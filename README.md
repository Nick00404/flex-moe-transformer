# MoE Transformer (Mixture of Experts)

A modern Transformer architecture with Mixture of Experts (MoE) integration, built on PyTorch. This implementation supports flexible routing strategies, expert balancing, and seamless integration with standard Transformer components.

## Features

- **Mixture of Experts (MoE)**: Dynamic routing to specialized expert networks
- **Flexible Architecture**: Can operate as standard Transformer or MoE model
- **Advanced Routing**: Top-k routing with group-level selection and bias balancing
- **Load Balancing**: Comprehensive auxiliary loss system for expert utilization
- **Warmup Support**: Gradual MoE activation during training
- **Debug Integration**: Built-in debugging capabilities via DebugMixin
- **Compatibility**: Maintains compatibility with standard Llama architecture

## Installation

```bash
pip install torch
# Clone repository and install dependencies
```

## Quick Start

```python
from moe_transformer import MoEModelConfig, TransformerForCausalLM

# Initialize model configuration
config = MoEModelConfig(
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=32,
    vocab_size=32000,
    use_moe=True,
    n_routed_experts=8,
    num_experts_per_tok=2
)

# Create model
model = TransformerForCausalLM(config)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    step=training_step  # For dynamic capacity adjustment
)
```

## Configuration

### Core Parameters
```python
MoEModelConfig(
    hidden_size=2048,              # Model dimension
    intermediate_size=8192,        # FFN hidden size
    num_hidden_layers=16,          # Number of transformer layers
    num_attention_heads=32,        # Attention heads
    vocab_size=32000,              # Vocabulary size
    rms_norm_eps=1e-05,            # Normalization epsilon
)
```

### MoE-Specific Parameters
```python
MoEModelConfig(
    use_moe=True,                   # Enable MoE
    n_routed_experts=8,            # Number of expert networks
    n_shared_experts=1,            # Shared experts (always active)
    num_experts_per_tok=2,         # Experts per token
    moe_intermediate_size=8192,    # Expert hidden size
    
    # Capacity management
    min_expert_capacity=128,
    max_expert_capacity=800,
    expert_capacity_factor=1.0,
    
    # Training dynamics
    moe_warmup_steps=100,          # Gradual MoE activation
    aux_loss_ramp_steps=100,       # Aux loss scaling
    aux_loss_coef=0.5,             # Load balancing coefficient
)
```

### Routing Parameters
```python
MoEModelConfig(
    n_group=2,                     # Routing groups
    topk_group=2,                  # Top groups to select from
    norm_topk_prob=True,           # Normalize expert weights
    
    # Advanced routing
    semantic_routing=False,        # Semantic-aware routing
    use_null_expert=True,          # Handle overflow tokens
    learned_token_importance=True, # Dynamic token prioritization
    use_bias_balancing=True,       # Expert utilization balancing
)
```

## Architecture

### Components

1. **TopkRouter**: Intelligent expert selection with:
   - Group-level routing
   - Bias balancing
   - Dynamic capacity adjustment
   - Periodic router reset

2. **ExpertProcessor**: Handles expert execution with:
   - Token importance weighting
   - Capacity-aware routing
   - Overflow handling

3. **FlexibleFFN**: Dual-path architecture:
   - Standard FFN path (always available)
   - MoE path (conditional)
   - Learnable combination weighting

4. **DecoderLayer**: Complete transformer layer with:
   - Self-attention (Grouped Query Attention)
   - Flexible FFN/MoE processing
   - Residual connections

### Key Features

- **Gradual Activation**: MoE warms up over specified steps
- **Dynamic Capacity**: Expert capacity adjusts during training
- **Load Balancing**: Comprehensive auxiliary loss system
- **Debug Integration**: Extensive logging and monitoring
- **Memory Efficiency**: Optimized expert utilization

## Usage Examples

### Basic Inference
```python
model = TransformerForCausalLM(config)
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits
```

### Training with MoE
```python
for step, batch in enumerate(dataloader):
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        step=step  # Pass step for dynamic adjustments
    )
    
    loss = outputs.loss
    aux_loss = outputs.aux_loss
    
    # Monitor expert usage
    expert_usage = outputs.expert_usage
    imbalances = outputs.imbalances
```

### Custom Routing
```python
# Custom router configuration
config = MoEModelConfig(
    scoring_func="softmax",        # sigmoid, softmax, or identity
    topk_method="noaux_tc",        # Routing strategy
    router_scaling_factor=1.0,     # Output scaling
)
```

## Advanced Features

### Expert Monitoring
```python
# Access expert utilization statistics
outputs = model(input_ids, output_expert_usage=True)
expert_counts = outputs.expert_usage  # List of expert usage per layer
imbalances = outputs.imbalances      # Load balancing metrics
```

### Dynamic Adjustments
```python
# Model automatically adjusts based on training step:
# - Expert capacity increases gradually
# - Auxiliary loss scales up
# - Router parameters reset periodically
```

### Debug Integration
```python
# Enable debug logging
model.set_debug(True)

# Monitor internal states
# (requires DebugMixin integration)
```

## Performance Considerations

### Memory Usage
- MoE models require more memory for expert parameters
- Use `min_expert_capacity` and `max_expert_capacity` to control memory footprint
- Consider gradient checkpointing for large models

### Training Stability
- Start with `moe_warmup_steps` to stabilize training
- Monitor `aux_loss` for expert balancing
- Use `router_reset_step` to prevent router collapse

## Best Practices

1. **Start Small**: Begin with few experts and gradually increase
2. **Monitor Utilization**: Check expert usage patterns regularly
3. **Balance Capacity**: Adjust `expert_capacity_factor` based on dataset
4. **Warmup Period**: Use sufficient `moe_warmup_steps` (100-1000)
5. **Regular Reset**: Enable `router_reset_step` for stability

## File Structure

```
moe_transformer/
├── __init__.py
├── core.py              # Base components (SwiGLUFFN, GroupedQueryAttention)
├── debug_mixin.py       # Debug utilities
├── llama.py             # Llama compatibility layer
└── moe_transformer.py   # Main implementation
```

## Dependencies

- PyTorch >= 1.9.0
- Python >= 3.7
- (Optional) DebugMixin for advanced monitoring

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{moe_transformer,
  title = {MoE Transformer Implementation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/moe-transformer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and issues:
- Open an issue on GitHub
- Check the examples directory for usage patterns
- Review the configuration documentation

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
