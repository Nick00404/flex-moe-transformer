import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
from core import MoEModelConfig, SwiGLU, RMSNorm, Attention

class SwiGLUMoE(nn.Module):
    """
    A minimal MoE layer that replaces the standard SwiGLU.
    For n_routed_experts=1, it should behave identically to the original after scaling.
    """
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
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

from safetensors.torch import save_file

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
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rope_type="llama3",
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
        rope_partial_ratio=1.0,
        sliding_window=None,
        dropout=0.0,
        use_lora=False,
    )

    # MoE config 
    moe_config = MoEModelConfig(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rope_type="llama3",
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
        rope_partial_ratio=1.0,
        sliding_window=None,
        dropout=0.0,
        use_lora=False,

        # MoE-specific params
        use_moe=True,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=4864,
        num_experts_per_tok=2,
        router_scaling_factor=1.0,
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

