from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass

    
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

@dataclass
class MoEModelConfig:
    # -----------------------------
    # Core parameters 
    # -----------------------------
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

    # -----------------------------
    # MoE (Mixture of Experts) parameters
    # -----------------------------
    use_moe: bool =False
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    n_group: int = 2
    topk_group: int = 1

class Attention(nn.Module):
    """
    Multi-head attention layer with support for GQA (grouped-query attention).
    Compatible with LLaMA/Mistral style configs (num_heads, num_kv_heads).
    """

    def __init__(self, config, layer_idx: int):
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
    ):
        bsz, seq_len, _ = x.size()

        # Project Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose for attention computation: [bsz, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k/v heads if num_kv_heads < num_heads (GQA)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (causal mask is [1,1,seq_len,seq_len])
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back to [bsz, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return (attn_output, attn_probs) if output_attentions else (attn_output, None)

class SwiGLU(nn.Module):
    """
    SwiGLU MLP block with down/up projection + gate.
    Implements:
        y = down_proj( swish(gate_proj(x)) * up_proj(x) )
    where swish(x) = x * sigmoid(x).
    """
    def __init__(self, config, bias: bool = False):
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
    """
    A minimal MoE layer that replaces the standard SwiGLU.
    For n_routed_experts=1, it should behave identically to the original after scaling.
    """
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Create N identical experts
        self.experts = nn.ModuleList([
            SwiGLU(config, bias=False) for _ in range(self.num_experts)
        ])

        # The router (gate) is a new linear layer that must be trained
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # Get routing logits: [batch_size, seq_len, num_experts]
        router_logits = self.router(x)
        
        if self.num_experts == 1:
            # Bypass routing and use expert 0 directly
            return self.experts[0](x)
        
        # For multiple experts, use the router
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # We need to scale the routing weights to account for the fact that we're using multiple experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        final_output = torch.zeros_like(x)
        
        # Calculate weighted sum of expert outputs
        for expert_idx, expert in enumerate(self.experts):
            # Create a mask for tokens that are assigned to this expert
            # We need to check all top-k positions for this expert
            expert_mask = (selected_experts == expert_idx)
            
            # If any token in the batch uses this expert, compute its output
            if expert_mask.any():
                # Get the indices where this expert is used
                batch_indices, seq_indices, topk_indices = torch.where(expert_mask)
                
                # Get the corresponding routing weights
                current_routing_weights = routing_weights[batch_indices, seq_indices, topk_indices]
                
                # Compute expert output for the relevant tokens
                expert_input = x[batch_indices, seq_indices]
                expert_output = expert(expert_input)
                
                # Add weighted expert output to final result
                for i, (b, s) in enumerate(zip(batch_indices, seq_indices)):
                    final_output[b, s] += current_routing_weights[i] * expert_output[i]
        
        return final_output
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, config: MoEModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Choose between standard MLP and MoE MLP
        if config.use_moe:
            # Use the new MoE layer
            self.mlp = SwiGLUMoE(config)
        else:
            # Use the original SwiGLU
            self.mlp = SwiGLU(config, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self attention
        residual = x
        x = self.input_layernorm(x)
        attn_output, _ = self.self_attn(x, attention_mask, position_ids, output_attentions=False)
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: MoEModelConfig,
    ):
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
                layer_idx=i  # Pass layer index
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
    ):
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
    
    def _create_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def load_tokenizer_from_json(tokenizer_path: str):
    """Load tokenizer from tokenizer.json file"""
    try:
        # Check if file exists
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        # Load using Hugging Face's tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to character-level tokenization")
        return None


from safetensors import safe_open

def load_model_from_safetensors(
    model_path: str,
    config: MoEModelConfig,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
) -> Transformer:
    """Load model weights from a safetensors file into a Transformer."""
    # Normalize device to string for safe_open
    if isinstance(device, torch.device):
        device_str = device.type  # "cuda" or "cpu"
    else:
        device_str = str(device)

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

            module = model
            for submodule in module_path.split("."):
                if hasattr(module, submodule):
                    module = getattr(module, submodule)
                elif submodule.isdigit():  # support numeric indices (e.g. layers.0.attn)
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


def copy_dense_to_moe(dense_model, moe_model):
    """
    Copies weights from a dense Transformer to an MoE Transformer.
    For experts: copies weights without scaling.
    For router: leaves it with random initialization.
    For all other layers: direct copy.
    """
    dense_sd = dense_model.state_dict()
    moe_sd = moe_model.state_dict()

    new_state_dict = {}
    num_experts = moe_model.config.n_routed_experts

    for key in moe_sd.keys():
        if 'router' in key:
            # Router weights are new - skip copying, keep random init
            print(f"Keeping random init for router weight: {key}")
            new_state_dict[key] = moe_sd[key]
        elif 'experts' in key:
            # This is an expert weight - copy directly without scaling
            parts = key.split('.')
            expert_idx_pos = parts.index('experts')
            layer_idx = parts[expert_idx_pos - 2]
            expert_idx = parts[expert_idx_pos + 1]
            param_name = parts[-1]
            expert_layer_name = parts[-2]

            # Find the corresponding dense key
            dense_key = f"layers.{layer_idx}.mlp.{expert_layer_name}.{param_name}"
            
            if dense_key in dense_sd:
                # Copy without scaling - scaling will be handled in forward pass
                new_state_dict[key] = dense_sd[dense_key]
                print(f"Copying: {dense_key} -> {key}")
            else:
                print(f"Warning: Could not find dense parameter {dense_key}")
                new_state_dict[key] = moe_sd[key]
        else:
            # Copy all other weights directly
            if key in dense_sd:
                new_state_dict[key] = dense_sd[key]
            else:
                print(f"Warning: Could not find dense parameter {key}")
                new_state_dict[key] = moe_sd[key]

    # Load the new state dict into the MoE model
    moe_model.load_state_dict(new_state_dict, strict=False)
    return moe_model

def save_moe_model(model, path="moe.safetensors"):
    """
    Saves a Transformer model to safetensors format,
    breaking shared weights to avoid safetensors RuntimeError.
    """
    state_dict = model.state_dict()

    # Detect and clone tied weights
    if "lm_head.weight" in state_dict and "embed_tokens.weight" in state_dict:
        if state_dict["lm_head.weight"].data_ptr() == state_dict["embed_tokens.weight"].data_ptr():
            print("🔧 Cloning lm_head.weight to avoid shared-memory issue...")
            state_dict = {k: v.clone() if k == "lm_head.weight" else v for k, v in state_dict.items()}

    save_file(state_dict, path)
    print(f"✅ MoE model saved to {path}")


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


# ===========================
# Example usage
# ===========================
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
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dense model
    print("Loading original dense model...")
    dense_model = load_model_from_safetensors("model.safetensors", dense_config, device=device)

    # Create MoE model
    print("Creating MoE model...")
    moe_model = Transformer(moe_config).to(device)

    # Copy weights
    print("Copying weights from dense to MoE...")
    moe_model = copy_dense_to_moe(dense_model, moe_model)

    # Save MoE model to disk
    save_moe_model(moe_model, "moe.safetensors")

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

    print("Reloading MoE model from moe.safetensors...")
    moe_model_reloaded = load_model_from_safetensors("moe.safetensors", moe_config, device=device)
    
    # Re-tie weights after loading
    if hasattr(moe_model_reloaded, "lm_head") and hasattr(moe_model_reloaded, "embed_tokens"):
        moe_model_reloaded.lm_head.weight = moe_model_reloaded.embed_tokens.weight
        print("🔗 Re-tied lm_head.weight to embed_tokens.weight")
    

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer_from_json("tokenizer.json")

    # Generate text
    print("Running generation...")
    output = generate_text(moe_model_reloaded, tokenizer, "The future of AI is", max_new_tokens=50)
    print("=== Generated Text ===")
    print(output)
