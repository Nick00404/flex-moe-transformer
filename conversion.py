"""
Mixture of Experts (MoE) Transformer implementation with weight conversion utilities.

This module provides:
- Transformer architecture with optional MoE layers
- Utilities to convert dense models to MoE models
- Safetensors integration for model loading/saving
"""

from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer equivalent to T5LayerNorm."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@dataclass
class MoEModelConfig:
    """Configuration for MoE Transformer model."""
    
    # Core parameters
    head_dim: int = 64
    hidden_size: int = 896
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-05
    vocab_size: int = 151936
    pad_token_id: int = 151643

    # MoE (Mixture of Experts) parameters
    use_moe: bool = False
    n_routed_experts: int = 4
    null_experts: int = 1
    num_experts_per_tok: int = 2
    n_group: int = 2
    topk_group: int = 1


class Attention(nn.Module):
    """
    Multi-head attention layer with support for GQA (grouped-query attention).
    Compatible with LLaMA/Mistral style configs.
    """

    def __init__(self, config: MoEModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.size()

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose for attention computation: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k/v heads if num_kv_heads < num_heads (GQA)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (causal mask is [1, 1, seq_len, seq_len])
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return (attn_output, attn_probs) if output_attentions else (attn_output, None)


class SwiGLU(nn.Module):
    """
    SwiGLU MLP block with down/up projection + gate.
    
    Implements: y = down_proj(swish(gate_proj(x)) * up_proj(x))
    where swish(x) = x * sigmoid(x).
    """
    
    def __init__(self, config: MoEModelConfig, bias: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Standard naming: gate, up, and down projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up  # elementwise product
        return self.down_proj(x)


class SwiGLUMoE(nn.Module):
    """Mixture of Experts layer that replaces standard SwiGLU with multiple experts."""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Projectors for expert selection (optional)
        self.importance_projector = nn.Linear(config.hidden_size, 1)
        self.semantic_projector = nn.Linear(config.hidden_size, config.hidden_size)

        # Create N identical experts
        self.experts = nn.ModuleList([SwiGLU(config, bias=False) for _ in range(self.num_experts)])

        # Router (gate) is a trainable linear layer
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get routing logits: [batch_size, seq_len, num_experts]
        router_logits = self.router(x)

        if self.num_experts == 1:
            # Bypass routing for single expert
            return self.experts[0](x)

        # For multiple experts, use the router
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_output = torch.zeros_like(x)

        # Calculate weighted sum of expert outputs
        for expert_idx, expert in enumerate(self.experts):
            # Create mask for tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx)

            # Process tokens using this expert
            if expert_mask.any():
                # Get indices where this expert is used
                batch_indices, seq_indices, topk_indices = torch.where(expert_mask)

                # Get corresponding routing weights
                current_routing_weights = routing_weights[batch_indices, seq_indices, topk_indices]

                # Compute expert output for relevant tokens
                expert_input = x[batch_indices, seq_indices]
                expert_output = expert(expert_input)

                # Add weighted expert output to final result
                for i, (batch_idx, seq_idx) in enumerate(zip(batch_indices, seq_indices)):
                    final_output[batch_idx, seq_idx] += current_routing_weights[i] * expert_output[i]

        return final_output


class TransformerBlock(nn.Module):
    """Transformer block with optional MoE MLP layer."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        config: MoEModelConfig,
        layer_idx: int
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Choose between standard MLP and MoE MLP
        if config.use_moe:
            self.mlp = SwiGLUMoE(config)
        else:
            self.mlp = SwiGLU(config, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention with residual connection
        residual = x
        x = self.input_layernorm(x)
        attn_output, _ = self.self_attn(x, attention_mask, position_ids, output_attentions=False)
        x = residual + attn_output

        # MLP with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Transformer(nn.Module):
    """Transformer model with optional MoE layers."""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.max_position_embeddings = config.max_position_embeddings

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.intermediate_size,
                config,
                layer_idx=i
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Create position_ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, input_ids.device)

        # Embed tokens
        x = self.embed_tokens(input_ids)

        # Forward through layers
        for layer in self.layers:
            x = layer(x, position_ids, attention_mask)

        # Final normalization
        x = self.norm(x)

        # Compute logits
        logits = self.lm_head(x)

        return logits

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def load_model_from_safetensors(
    model_path: Union[str, Path],
    config: MoEModelConfig,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
) -> Transformer:
    """Load model weights from a safetensors file into a Transformer."""
    # Normalize device to string for safe_open
    device_str = device.type if isinstance(device, torch.device) else str(device)

    # Initialize model on the requested device
    model = Transformer(config).to(device_str)

    # Load weights from safetensors
    with safe_open(model_path, framework="pt", device=device_str) as f:
        for key in f.keys():
            # Map parameter names (strip "model." prefix if present)
            param_key = key[len("model."):] if key.startswith("model.") else key

            # Resolve module path and parameter name
            try:
                module_path, param_name = param_key.rsplit(".", 1)
            except ValueError:
                print(f"⚠️ Skipping key {param_key}, unexpected format")
                continue

            # Navigate to target module
            module = model
            for submodule in module_path.split("."):
                if hasattr(module, submodule):
                    module = getattr(module, submodule)
                elif submodule.isdigit():  # Support numeric indices (e.g., layers.0.attn)
                    module = module[int(submodule)]
                else:
                    print(f"⚠️ Could not find submodule {submodule} in path {module_path}")
                    module = None
                    break

            if module is None:
                continue

            # Copy tensor into model parameter
            try:
                tensor = f.get_tensor(key)
                getattr(module, param_name).data.copy_(tensor)
            except AttributeError:
                print(f"⚠️ Could not set parameter {param_name} for module {module_path}")

    return model


def copy_dense_to_moe(dense_model: Transformer, moe_model: Transformer) -> Transformer:
    """
    Copy weights from a dense Transformer to an MoE Transformer.
    
    Expert weights are copied without scaling, router weights are randomly initialized,
    and all other weights are copied directly. Optimized for GPU memory efficiency.
    """
    dense_sd = dense_model.state_dict()
    moe_sd = moe_model.state_dict()

    new_state_dict = {}
    num_experts = getattr(moe_model.config, "n_routed_experts", None)

    # First, copy all non-expert weights
    print("Copying non-expert weights...")
    for key in moe_sd.keys():
        if any(routing_key in key for routing_key in [
            "router", "importance_projector", "semantic_projector", 
            "embedding_projection", "attention_pooling"
        ]):
            print(f"Keeping random init for routing/retrieval weight: {key}")
            new_state_dict[key] = moe_sd[key]
        elif "expert" not in key:  # Non-expert weights
            if key in dense_sd:
                new_state_dict[key] = dense_sd[key]
            else:
                print(f"Warning: Could not find dense parameter {key}")
                new_state_dict[key] = moe_sd[key]

    # Load non-expert weights first to free up memory
    moe_model.load_state_dict(new_state_dict, strict=False)

    # Clean up dense model to free memory for expert copying
    del dense_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Copying expert weights...")
    # Process expert weights
    for key in moe_sd.keys():
        if "expert" in key:  # Match any expert-related keys
            parts = key.split(".")
            try:
                # Find the expert index position
                expert_idx_pos = max(i for i, p in enumerate(parts) if "expert" in p)

                layer_idx = parts[expert_idx_pos - 2]
                expert_layer_name = parts[-2]
                param_name = parts[-1]

                dense_key = f"layers.{layer_idx}.mlp.{expert_layer_name}.{param_name}"

                # Load the dense parameter
                if dense_key in dense_sd:
                    tensor = dense_sd[dense_key]
                    if torch.cuda.is_available():
                        tensor = tensor.cuda()

                    # Get target parameter and copy data
                    module = moe_model
                    for submodule in key.split(".")[:-1]:
                        if hasattr(module, submodule):
                            module = getattr(module, submodule)
                        elif submodule.isdigit():
                            module = module[int(submodule)]

                    getattr(module, key.split(".")[-1]).data.copy_(tensor)
                    print(f"Copied: {dense_key} -> {key}")
                else:
                    print(f"Warning: Could not find dense parameter {dense_key}")

            except Exception as e:
                print(f"⚠️ Skipping {key} due to parsing error: {e}")

    return moe_model


def save_moe_model(model: Transformer, path: Union[str, Path] = "model.safetensors") -> None:
    """Save a Transformer model to safetensors format, handling shared weights."""
    state_dict = model.state_dict()

    # Detect and clone tied weights to avoid safetensors RuntimeError
    if ("lm_head.weight" in state_dict and "embed_tokens.weight" in state_dict and
        state_dict["lm_head.weight"].data_ptr() == state_dict["embed_tokens.weight"].data_ptr()):
        print("🔧 Cloning lm_head.weight to avoid shared-memory issue...")
        state_dict = {k: v.clone() if k == "lm_head.weight" else v for k, v in state_dict.items()}

    save_file(state_dict, path)
    print(f"✅ MoE model saved to {path}")


def main() -> None:
    """Example usage: Convert dense model to MoE model."""
    # Dense model config
    dense_config = MoEModelConfig(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
    )

    # MoE config
    moe_config = MoEModelConfig(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,

        # MoE-specific params
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dense model
    print("Loading original dense model...")
    dense_model = load_model_from_safetensors("dense_model.safetensors", dense_config, device=device)

    # Create MoE model
    print("Creating MoE model...")
    moe_model = Transformer(moe_config).to(device)

    # Copy weights with memory optimization
    print("Copying weights from dense to MoE...")
    moe_model = copy_dense_to_moe(dense_model, moe_model)

    # Save MoE model to disk
    save_moe_model(moe_model, "model.safetensors")


if __name__ == "__main__":
    main()
