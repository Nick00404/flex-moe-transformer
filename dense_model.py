import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
from core import MoEModelConfig, SwiGLU, RMSNorm, Attention

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, config: MoEModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        # Use SwiGLU instead of MLP
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


class TextGenerator:
    def __init__(self, model: Transformer, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        # Tokenize input using the tokenizer
        if self.tokenizer:
            # Use the proper tokenizer encoding
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            # Fallback to simple character-level tokenization
            input_ids = torch.tensor([[ord(c) for c in prompt]], device=self.device)
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions - only use the last 2048 tokens for context
                # (to avoid exceeding max sequence length)
                context = input_ids[:, -2048:] if input_ids.shape[1] > 2048 else input_ids
                logits = self.model(context)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_value = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_value)[0][-1]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        # Decode output
        if self.tokenizer:
            return self.tokenizer.decode(input_ids[0])
        else:
            return ''.join([chr(token) for token in input_ids[0].cpu().numpy()])


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


def load_model_from_safetensors(
    model_path: str,
    config: MoEModelConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Transformer:
    """Load model weights from safetensors file"""
    # Initialize model
    model = Transformer(config).to(device)
    
    # Load weights from safetensors
    with safe_open(model_path, framework="pt", device=device) as f:
        for key in f.keys():
            # Map parameter names if needed
            param_name = key
            if key.startswith("model."):
                param_name = key[len("model."):]
            
            # Get the target parameter
            module_path, param_name = param_name.rsplit(".", 1)
            module = model
            for submodule in module_path.split("."):
                if hasattr(module, submodule):
                    module = getattr(module, submodule)
                elif submodule.isdigit():
                    module = module[int(submodule)]
                else:
                    print(f"Warning: Could not find submodule {submodule} in path {module_path}")
                    continue
            
            # Load the parameter
            try:
                param_data = f.get_tensor(key)
                getattr(module, param_name).data.copy_(param_data)
            except AttributeError:
                print(f"Warning: Could not set parameter {param_name} for module {module_path}")
    
    return model


# Example usage
if __name__ == "__main__":
    config = MoEModelConfig(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=32768,
        # Add RoPE parameters
        rope_theta=1000000.0,
        rope_type="llama3",
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
        # Add new parameters for GQA
        rope_partial_ratio=1.0,  # Full rotary embeddings by default
        sliding_window=None,     # No sliding window by default
        dropout=0.0,            # No dropout by default
        use_lora=False          # LoRA configuration
    )
    
    # Load the tokenizer
    tokenizer = load_tokenizer_from_json("tokenizer.json")
    
    # Load the model
    model = load_model_from_safetensors("model.safetensors", config)
    
    # Create generator with tokenizer
    generator = TextGenerator(model, tokenizer)
    
    # Generate text
    prompt = "<|im_start|>user\nWho are you? What is Qwen?<|im_end|>\n<|im_start|>assistant\n"
    generated_text = generator.generate(prompt, max_length=50)
    
    print("Generated text:")
    print(generated_text)