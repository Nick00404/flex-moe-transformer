from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
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
    hidden_size: int = 896
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-05
    vocab_size: int = 151936
    pad_token_id: int = 151643
    dropout: float = 0.0

    # MoE (Mixture of Experts) parameters
    use_moe: bool = False
    n_routed_experts: int = 4
    num_experts_per_tok: int = 2
    n_group: int = 2
    topk_group: int = 1

class MLP(nn.Module):
    """
    MLP block with down/up projection + gate.
    
    Implements: y = down_proj(swish(gate_proj(x)) * up_proj(x))
    where swish(x) = x * sigmoid(x).
    """
    
    def __init__(self, config: MoEModelConfig, bias: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        dropout = config.dropout
        
        # Use standard nn.Linear layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = self.act_fn(gate) * up
        dropped = self.dropout(activated)
        output = self.down_proj(dropped)
        return output


class MoE(nn.Module):
    """Mixture of Experts layer that replaces standard MLP with multiple experts."""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Projectors for expert selection
        self.importance_projector = nn.Linear(config.hidden_size, 1)
        self.semantic_projector = nn.Linear(config.hidden_size, config.hidden_size)
        bias_tensor = torch.zeros(config.n_routed_experts)
        self.e_score_correction_bias = nn.Parameter(bias_tensor)

        # Create N identical experts
        self.experts = nn.ModuleList([MLP(config, bias=False) for _ in range(self.n_routed_experts)])

        # Number of special experts
        self.shared_expert_count = 1
        self.null_expert_count = 1

        # Regular experts
        self.experts = nn.ModuleList([MLP(config, bias=False) for _ in range(self.n_routed_experts)])

        # Special experts
        self.shared_expert = nn.ModuleList([MLP(config, bias=False) for _ in range(self.shared_expert_count)])
        self.null_expert = nn.ModuleList([MLP(config, bias=False) for _ in range(self.null_expert_count)])

        # Router (gate) is a trainable linear layer
        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute router logits
        router_logits = self.gate(x)

        # Fast path: single expert (no routing needed)
        if self.n_routed_experts == 1:
            return self.experts[0](x)

        # Compute routing weights
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Select top-k experts per token
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Flatten for processing
        x_reshaped = x.view(-1, x.size(-1))
        selected_experts_reshaped = selected_experts.view(-1, self.num_experts_per_tok)
        routing_weights_reshaped = routing_weights.view(-1, self.num_experts_per_tok)

        # Prepare output
        final_output = torch.zeros_like(x_reshaped)

        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Boolean mask of tokens going to this expert
            mask = (selected_experts_reshaped == expert_idx)
            if not mask.any():
                continue

            # Indices of tokens routed here
            token_idx = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue

            # Gather input for this expert
            current_input = x_reshaped[token_idx]

            # Gather weights safely using mask
            current_weights = (routing_weights_reshaped[token_idx] * mask[token_idx]).sum(dim=1, keepdim=True)

            # Expert forward
            expert_output = expert(current_input)

            # Weighted sum into output
            final_output[token_idx] += current_weights * expert_output

        return final_output.view_as(x)



class Attention(nn.Module):
    def __init__(self, config: MoEModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = self.num_attention_heads // self.num_kv_heads
        self.dropout = config.dropout

        # Projection layers
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    def _prepare_qkv(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        bsz, seq_len, _ = hidden_states.shape

        # Project to Q/K/V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV heads if needed
        if self.num_kv_heads != self.num_attention_heads:
            repeat_factor = self.num_attention_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        return q, k, v
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        if position_ids is None:
            raise ValueError("position_ids must be provided")
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.to(hidden_states.device).long()

        q, k, v = self._prepare_qkv(hidden_states, position_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q.dtype, device=q.device)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        attn_probs = None
        if output_attentions:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_scores += attention_mask
            attn_probs = F.softmax(attn_scores, dim=-1)

        if attn_output is None:
            raise RuntimeError("Attention output is None — check dtype/device issues!")

        return attn_output, attn_probs
    
class DecoderLayer(nn.Module):
    """DecoderLayer with optional MoE MLP layer."""
    
    def __init__(
        self,
        hidden_size: int,
        config: MoEModelConfig,
        layer_idx: int
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Choose between standard MLP and MoE MLP
        if config.use_moe:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config, bias=False)

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
            DecoderLayer(
                config.hidden_size,
                config,
                layer_idx=i
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        return mask.unsqueeze(0).unsqueeze(0)


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
            "gate", "importance_projector", "semantic_projector", "e_score_correction_bias",
        ]):
            print(f"Keeping random init for routing/retrieval weight: {key}")
            new_state_dict[key] = moe_sd[key]
        elif "expert" not in key:  # Non-expert weights
            if key in dense_sd:
                new_state_dict[key] = dense_sd[key]
                print(f"Copied: {key}")
            else:
                print(f"Warning: Could not find dense parameter {key}")
                new_state_dict[key] = moe_sd[key]

    # Load non-expert weights first to free up memory
    moe_model.load_state_dict(new_state_dict, strict=False)
    
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
    """Save a Transformer model to safetensors format, ignoring lm_head if it shares weights."""
    state_dict = model.state_dict().copy()

    # Remove lm_head if it exists (to avoid duplicating tied weights)
    if "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
        print("⚠️ Ignored lm_head.weight while saving (tied to embed_tokens)")

    save_file(state_dict, path)
    print(f"✅ MoE model saved to {path}")

if __name__ == "__main__":
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
        n_routed_experts=6,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dense model directly on GPU (or selected device)
    print("Loading original dense model on device...")
    dense_model = load_model_from_safetensors("model.safetensors", dense_config, device=device).half()
    print(f"Dense model loaded on {device}")

    # Create MoE model on GPU
    print("Creating MoE model and copying weights...")
    moe_model = copy_dense_to_moe(dense_model, Transformer(moe_config).to(device).half())
    print("MoE model Ready...")

    save_moe_model(moe_model, "moe.safetensors")

# Last cell

def load_tokenizer_from_json(tokenizer_path: str):
    """Load tokenizer from tokenizer.json file"""
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    return tokenizer

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)[:, -1, :]  # last token logits
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k > 0:
                # Top-k filtering
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, indices, values)
                logits = mask
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

if __name__ == "__main__":
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
        n_routed_experts=6,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dense model
    print("Loading original dense model...")
    dense_model = load_model_from_safetensors("model.safetensors", dense_config, device=device).half()

    # Create MoE model
    print("Loading MoE model...")
    moe_model = load_model_from_safetensors("moe.safetensors", moe_config, device=device).half()

    # tie weights
    if hasattr(moe_model, "lm_head") and hasattr(moe_model, "embed_tokens"):
        moe_model.lm_head.weight = moe_model.embed_tokens.weight
        print("🔗 tied MoE lm_head.weight to embed_tokens.weight")

    # tie lm_head
    if hasattr(dense_model, "lm_head") and hasattr(dense_model, "embed_tokens"):
        dense_model.lm_head.weight = dense_model.embed_tokens.weight
        print("🔗 tied Dense lm_head.weight to embed_tokens.weight")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer_from_json("tokenizer.json")

    # Generate text
    print("Running generation...")
    output = generate_text(moe_model, tokenizer, "The future of AI is", max_new_tokens=50)
    print("=== Generated Text ===")
    print(output)

    # Test equivalence
    print("Testing model equivalence...")
    test_input = torch.randint(0, 1000, (1, 10)).to(device)

    with torch.no_grad():
        dense_output = dense_model(test_input)
        moe_output = moe_model(test_input)

    similarity = torch.cosine_similarity(dense_output.flatten(), moe_output.flatten(), dim=0)
    print(f"Cosine similarity between dense and MoE outputs: {similarity.item():.8f}")

    if similarity.item() > 0.999:
        print("✅ SUCCESS: MoE model behaves identically to the dense model!")
    else:
        print("❌ FAILURE: Outputs are not similar enough.")

