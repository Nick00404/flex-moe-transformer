from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.modeling_outputs import CausalLMOutput
from modeling_custom_model import MoE, MoEOutput, RMSNorm, MLP, Attention, MoEModelConfig

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
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout
            )

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
        mlp_out = self.mlp(x)

        # Unwrap MoEOutput if MoE is being used
        if isinstance(mlp_out, MoEOutput):
            x = mlp_out.last_hidden_state
        else:
            x = mlp_out

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
    ) -> CausalLMOutput:
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

        # Return in HF-friendly format
        return CausalLMOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


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

                # Handle null_expert (no .0 in path)
                if "null_expert" in key:
                    dense_key = f"layers.{layer_idx}.mlp.{expert_layer_name}.{param_name}"
                else:
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

def load_tokenizer_from_json(tokenizer_path: str):
    """Load tokenizer from tokenizer.json file"""
    from transformers import PreTrainedTokenizerFast
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
            # Call model and grab logits from CausalLMOutput
            output = model(generated)
            logits = output.logits[:, -1, :]  # only last token

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
    dense_config = MoEModelConfig()

    # MoE config 
    moe_config = MoEModelConfig()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dense model directly on GPU (or selected device)
    print("Loading original dense model on device...")
    dense_model = load_model_from_safetensors("model.safetensors", dense_config, device=device)
    print(f"Dense model loaded on {device}")

    # Create MoE model on GPU
    print("Creating MoE model and copying weights...")
    moe_model = copy_dense_to_moe(dense_model, Transformer(moe_config).to(device))
    print("MoE model Ready...")

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

    # Set models to evaluation mode
    dense_model.eval()
    moe_model.eval()
    print("Models set to evaluation mode")

    # Generate text with both models for comparison
    print("Running generation with both models...")
    
    # MoE generation
    moe_output = generate_text(moe_model, tokenizer, "The future of AI is", max_new_tokens=50)
    print("=== MoE Generated Text ===")
    print(moe_output)
    
    # Dense generation for comparison
    dense_output = generate_text(dense_model, tokenizer, "The future of AI is", max_new_tokens=50)
    print("=== Dense Generated Text ===")
    print(dense_output)

    # Test equivalence with multiple inputs for better verification
    print("Testing model equivalence with multiple inputs...")
    
    # Create multiple test inputs
    test_inputs = [
        torch.randint(0, tokenizer.vocab_size, (1, 10)).to(device),
        torch.randint(0, tokenizer.vocab_size, (1, 5)).to(device),
        torch.randint(0, tokenizer.vocab_size, (1, 15)).to(device),
    ]
    
    similarities = []
    for i, test_input in enumerate(test_inputs):
        with torch.no_grad():
            dense_logits = dense_model(test_input).logits
            moe_logits = moe_model(test_input).logits
            
            # Calculate multiple similarity metrics
            cosine_sim = torch.cosine_similarity(dense_logits.flatten(), moe_logits.flatten(), dim=0)
            mse = torch.nn.functional.mse_loss(dense_logits, moe_logits)
            
            similarities.append((cosine_sim.item(), mse.item()))
            
            print(f"Test {i+1}: Cosine similarity = {cosine_sim.item():.8f}, MSE = {mse.item():.8f}")
    
    # Calculate average similarity
    avg_cosine_sim = sum(s[0] for s in similarities) / len(similarities)
    print(f"Average cosine similarity: {avg_cosine_sim:.8f}")

    if avg_cosine_sim > 0.9999:
        print("✅ SUCCESS: MoE model behaves identically to the dense model!")
    else:
        print("❌ FAILURE: Outputs are not similar enough.")
        
    save_moe_model(moe_model, "moe.safetensors")
