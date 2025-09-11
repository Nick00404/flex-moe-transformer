import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from debug_mixin import DebugMixin
from core import SwiGLUFFN, GroupedQueryAttention, LlamaRMSNorm
from llama import LlamaModel, LlamaForCausalLM


@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor  # Final transformer hidden states
    hidden_states: Optional[List[torch.Tensor]] = None  # Hidden states from all layers
    logits: torch.Tensor = None      # Final output logits (from LM head)
    loss: Optional[torch.Tensor] = None  # Total loss (cross-entropy + aux_loss)
    aux_loss: torch.Tensor = 0.0     # Sum of all auxiliary losses
    expert_usage: Optional[List[torch.Tensor]] = None  # Expert usage counts per layer
    router_logits: Optional[List[torch.Tensor]] = None  # Router logits per layer
    attentions: Optional[List[torch.Tensor]] = None
    imbalances: Optional[List[torch.Tensor]] = None

@dataclass
class DecoderLayerOutput:
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    aux_loss: torch.Tensor = 0.0
    expert_usage: Optional[torch.Tensor] = None


@dataclass
class MoEModelConfig:
    # -----------------------------
    # Core parameters
    # -----------------------------
    head_dim: int = 64
    hidden_size: int = 2048
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072
    num_attention_heads: int = 32
    num_hidden_layers: int = 16
    num_key_value_heads: int = 8

    rms_norm_eps: float = 1e-05
    use_cache: bool = False
    vocab_size: int = 128256
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: int = 128002

    # -----------------------------
    # RoPE scaling
    # -----------------------------
    rope_type: Optional[str] = "llama3"
    rope_theta: float = 500000.0
    factor: Optional[float] = 32.0
    original_max_position_embeddings: Optional[int] = 8192
    low_freq_factor: Optional[float] = 1.0
    high_freq_factor: Optional[float] = 4.0

    # -----------------------------
    # LoRA parameters
    # -----------------------------
    use_lora: bool = False
    q_lora_rank: int = 8          # Targeted rank for Q projections
    k_lora_rank: int = 4 
    v_lora_rank: int = 4         # Lower rank for K/V projections
    lora_alpha: float = 16.0      # Scaling factor (alpha = 2x max rank)
    lora_dropout: float = 0.1     # Regularization for LoRA layers
    lora_rank: int = 0            # Disabled (using per-head ranks)
    lora_scaling: Optional[float] = None  # Auto-scale via alpha/rank

    sliding_window: Optional[int] = 4096  # Local attention span
    rope_partial_ratio: float = 1.0

    # -----------------------------
    # MoE (Mixture of Experts) parameters
    # -----------------------------
    use_moe: bool = False
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 8192
    num_experts_per_tok: int = 2
    router_scaling_factor: float = 1.0
    n_group: int = 2
    topk_group: int = 2
    norm_topk_prob: bool = True

    # -----------------------------
    # Expert capacity and growth
    # -----------------------------
    min_expert_capacity: int = 128
    max_expert_capacity: int = 800
    expert_capacity_factor: float = 1.0
    capacity_ramp_rate: float = 0.05

    # -----------------------------
    # Warmup and training steps
    # -----------------------------
    moe_warmup_steps: int = 100
    aux_loss_ramp_steps: int = 100
    aux_loss_max_scale: float = 5.0
    router_reset_step: int = 100
    router_reset_noise_scale: float = 0.02

    # -----------------------------
    # Balancing and loss coefficients
    # -----------------------------
    aux_loss_coef: float = 0.5
    semantic_routing: bool = False           # Enable semantic-aware routing
    use_null_expert: bool = True             # Use null expert for overflow
    learned_token_importance: bool = True    # Enable learned token importance
    use_bias_balancing: bool = True
    bias_update_rate: float = 0.01

    # Output configuration
    output_attentions: bool = False
    output_hidden_states: bool = False

    dropout: float = 0.1
    initializer_range: float = 0.02
    
    divisibility_mode: str = "error"

    def __post_init__(self):
        """
        Normalize and validate configuration parameters right after initialization.
        This runs automatically for dataclasses.
        """
        # ---- Head dimension checks and defaults ----
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
    
        # ---- MoE-related validation ----
        if self.use_moe:
            if self.moe_intermediate_size is None:
                self.moe_intermediate_size = self.intermediate_size
            if self.n_routed_experts <= 0:
                raise ValueError("n_routed_experts must be positive when use_moe=True")
    
        # ---- Run divisibility and MoE invariants ----
        # Enforce/auto-fix divisibility for MHA
        if self.hidden_size % self.num_attention_heads != 0:
            if self.divisibility_mode == "error":
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})."
                )
            elif self.divisibility_mode == "auto_heads":
                new_heads = self._largest_divisor_le(self.hidden_size, self.num_attention_heads)
                if new_heads != self.num_attention_heads:
                    warnings.warn(
                        f"Adjusted num_attention_heads from {self.num_attention_heads} to {new_heads} "
                        f"to divide hidden_size={self.hidden_size}.",
                        RuntimeWarning,
                    )
                    self.num_attention_heads = new_heads
            elif self.divisibility_mode == "auto_d_model":
                new_hidden_size = math.ceil(self.hidden_size / self.num_attention_heads) * self.num_attention_heads
                if new_hidden_size != self.hidden_size:
                    warnings.warn(
                        f"Adjusted hidden_size from {self.hidden_size} to {new_hidden_size} "
                        f"to be divisible by num_attention_heads={self.num_attention_heads}.",
                        RuntimeWarning,
                    )
                    self.hidden_size = new_hidden_size
            else:
                raise ValueError(
                    f"Unknown divisibility_mode={self.divisibility_mode!r}"
                )

        # MoE invariants
        max_selectable = self.n_routed_experts + (1 if self.use_null_expert else 0)
        if self.num_experts_per_tok > max_selectable:
            warnings.warn(
                f"num_experts_per_tok ({self.num_experts_per_tok}) > selectable experts ({max_selectable}); "
                f"clamping to {max_selectable}.",
                RuntimeWarning,
            )
            self.num_experts_per_tok = max_selectable

        if self.min_expert_capacity > self.max_expert_capacity:
            warnings.warn(
                f"min_expert_capacity ({self.min_expert_capacity}) > max_expert_capacity "
                f"({self.max_expert_capacity}); swapping.",
                RuntimeWarning,
            )
            self.min_expert_capacity, self.max_expert_capacity = (
                self.max_expert_capacity,
                self.min_expert_capacity,
            )

    # ---------- helpers ----------
    @staticmethod
    def _largest_divisor_le(n: int, limit: int) -> int:
        for h in range(min(limit, n), 0, -1):
            if n % h == 0:
                return h
        return 1
    
class TopkRouter(nn.Module):
    """Top-k Router for Mixture of Experts (MoE) systems with simplified losses."""
    def __init__(
        self, 
        config: MoEModelConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.aux_loss_coef = config.aux_loss_coef

        # Dynamic scaling parameters
        self.aux_loss_ramp_steps = config.aux_loss_ramp_steps
        self.aux_loss_max_scale = config.aux_loss_max_scale

        # Validate configuration
        if self.n_routed_experts % self.n_group != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})"
            )
        if self.topk_group > self.n_group:
            raise ValueError(
                f"topk_group ({self.topk_group}) cannot exceed n_group ({self.n_group})"
            )

        # Configurable scoring and selection
        self.scoring_func = getattr(config, "scoring_func", "sigmoid")
        self.topk_method = getattr(config, "topk_method", "noaux_tc")

        # Routing weights
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), device=device, dtype=dtype)
        )

        # Optional trainable bias
        trainable_bias = getattr(config, "trainable_bias", False)
        bias_tensor = torch.zeros(self.n_routed_experts, device=device, dtype=dtype)
        if trainable_bias:
            self.e_score_correction_bias = nn.Parameter(bias_tensor)
        else:
            self.register_buffer("e_score_correction_bias", bias_tensor)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.weight.data.mul_(1 / math.sqrt(self.config.hidden_size))

    def update_biases(self, expert_counts: torch.Tensor):
        """Update biases based on expert utilization"""
        if not self.config.use_bias_balancing:
            return

        total_tokens = expert_counts.sum().float()
        target = total_tokens * self.top_k / self.n_routed_experts
        errors = target - expert_counts.float()

        self.e_score_correction_bias += (
            self.config.bias_update_rate * torch.sign(errors)
        )

    def score_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        elif self.scoring_func == "softmax":
            return torch.softmax(logits, dim=-1)
        elif self.scoring_func == "identity":
            return logits
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")

    @torch.no_grad()
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Expert selection"""
        if self.config.use_bias_balancing:
            scores_for_choice = scores + self.e_score_correction_bias.detach()
        else:
            scores_for_choice = scores + self.e_score_correction_bias

        # Stage 1: Group-level selection
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(self.topk_group, dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top groups
        _, group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Stage 2: Expert-level selection
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, -1, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), -torch.inf)

        # Select top-k experts
        _, topk_indices = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        return topk_indices

    def compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss for MoE routing (imbalance-based),
        with optional attention masking for padded tokens.
        """
        # Flatten logits if needed
        if router_logits.dim() != 2:
            router_logits = router_logits.view(-1, router_logits.size(-1))
    
        # Convert logits to probabilities
        router_probs = F.softmax(router_logits, dim=-1)
    
        # Mask of which experts are selected per token
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts).sum(dim=1).float()
    
        # Apply attention mask if provided
        if attention_mask is not None:
            flat_mask = attention_mask.view(-1, 1).float()
            expert_mask = expert_mask * flat_mask
            router_probs = router_probs * flat_mask
    
        # Fraction of tokens assigned per expert
        tokens_per_expert = expert_mask.sum(dim=0)
        total_tokens = tokens_per_expert.sum() + 1e-6
        fraction_tokens = tokens_per_expert / total_tokens
    
        # Fraction of router probability mass per expert
        router_prob_per_expert = router_probs.sum(dim=0)
        total_router_prob = router_prob_per_expert.sum() + 1e-6
        fraction_router_prob = router_prob_per_expert / total_router_prob
    
        # Imbalance-based auxiliary loss (standard GShard/Switch)
        imbalance = torch.abs(fraction_tokens - 1 / self.n_routed_experts).sum()
        aux_loss = imbalance * self.config.aux_loss_coef
    
        return aux_loss
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        step: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (step is not None and 
            self.config.router_reset_step > 0 and 
            step % self.config.router_reset_step == 0 and 
            step > 0 and 
            self.training):

            self.reset_parameters()
            with torch.no_grad():
                noise = torch.randn_like(self.weight) * self.config.router_reset_noise_scale
                self.weight.add_(noise)
                if isinstance(self.e_score_correction_bias, nn.Parameter):
                    bias_noise = torch.randn_like(self.e_score_correction_bias) * self.config.router_reset_noise_scale
                    self.e_score_correction_bias.add_(bias_noise)

        # Flatten hidden states
        hidden_states = hidden_states.view(-1, self.config.hidden_size)

        # Compute router logits
        router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight.to(hidden_states.device))
        router_logits = torch.clamp(router_logits, -3.0, 3.0)

        # Compute scores
        scores = self.score_fn(router_logits)

        # Select top-k experts
        topk_indices = self.get_topk_indices(scores)

        # Gather top-k weights
        topk_weights = scores.gather(1, topk_indices)

        # Normalize weights
        if self.norm_topk_prob and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Apply router scaling
        topk_weights = topk_weights * (self.config.router_scaling_factor / 2) 

        # Compute auxiliary loss during training
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            aux_loss = self.compute_aux_loss(
                router_logits,
                topk_indices,
                attention_mask=attention_mask
            )

            # Dynamic aux loss scaling
            if step is not None:
                aux_scale = min(
                    self.aux_loss_max_scale,
                    1.0 + (step / self.aux_loss_ramp_steps) * (self.aux_loss_max_scale - 1.0)
                )
                aux_loss = aux_loss * aux_scale

            # Update biases for balancing
            if self.config.use_bias_balancing:
                expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts).sum(dim=1)
                expert_counts = expert_mask.sum(dim=0)
                self.update_biases(expert_counts)

        return topk_indices, topk_weights, aux_loss, router_logits

    
class ExpertProcessor(DebugMixin):
    """Handles expert routing and processing logic"""
    def __init__(self, config: MoEModelConfig, experts: nn.ModuleList, 
                 shared_experts: nn.ModuleList, null_expert: Optional[nn.Module] = None):
        self.config = config
        self.experts = experts
        self.shared_experts = shared_experts
        self.null_expert = null_expert
        
    def calculate_expert_capacity(self, step: Optional[int] = None) -> int:
        """Dynamically calculate expert capacity based on training progress"""
        base_capacity = self.config.min_expert_capacity
        max_capacity = self.config.max_expert_capacity
        
        if step is not None:
            ramp = min(step * self.config.capacity_ramp_rate, 1.0)
            expert_capacity = int(base_capacity + ramp * (max_capacity - base_capacity))
            expert_capacity = min(expert_capacity, int(self.config.expert_capacity_factor * max_capacity))
        else:
            expert_capacity = base_capacity
            
        return expert_capacity
        
    def process_routed_experts(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, hidden_size = hidden_states.shape
        top_k = topk_indices.size(1)
        expert_capacity = self.calculate_expert_capacity(step)

        output = torch.zeros_like(hidden_states)
        expert_counts = torch.zeros(len(self.experts), device=hidden_states.device, dtype=torch.long)

        token_importance = torch.norm(hidden_states, dim=1)
        token_importance = token_importance / (token_importance.sum() + 1e-6)

        # Track unrouted tokens
        overflow_tokens = torch.zeros(batch_size, dtype=torch.bool, device=hidden_states.device)

        for k in range(top_k):
            expert_ids = topk_indices[:, k]
            weights = topk_weights[:, k]

            for expert_id in torch.unique(expert_ids):
                token_mask = (expert_ids == expert_id)
                token_idx = token_mask.nonzero(as_tuple=True)[0]

                if token_idx.numel() == 0 or expert_counts[expert_id] >= expert_capacity:
                    overflow_tokens[token_idx] = True
                    continue

                if token_idx.numel() > (expert_capacity - expert_counts[expert_id]):
                    importance_subset = token_importance[token_idx]
                    _, priority_idx = torch.topk(
                        importance_subset, k=expert_capacity - expert_counts[expert_id]
                    )
                    dropped_idx = torch.ones_like(token_idx, dtype=torch.bool)
                    dropped_idx[priority_idx] = False
                    overflow_tokens[token_idx[dropped_idx]] = True
                    token_idx = token_idx[priority_idx]

                tokens = hidden_states.index_select(0, token_idx)
                expert_out = self.experts[expert_id](tokens)
                weighted_out = expert_out * weights[token_idx].unsqueeze(-1)
                output.index_add_(0, token_idx, weighted_out)
                expert_counts[expert_id] += token_idx.numel()

        #self._logger.debug(f"per layer experts: {expert_counts.tolist()}")
        return output, expert_counts


class MoE(DebugMixin, nn.Module):
    """Mixture of Experts module with debug instrumentation"""
    
    def __init__(self, config: MoEModelConfig):
        nn.Module.__init__(self)
        DebugMixin.__init__(self)
        
        self.config = config
        self.enabled = config.use_moe

        # Token importance projection
        if config.learned_token_importance:
            self.importance_projector = nn.Linear(config.hidden_size, 1)
            nn.init.normal_(self.importance_projector.weight, 0, 0.02)
            nn.init.zeros_(self.importance_projector.bias)
        else:
            self.importance_projector = None

        # Semantic routing projection
        if config.semantic_routing:
            self.semantic_projector = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.normal_(self.semantic_projector.weight, 0, 0.02)
            nn.init.zeros_(self.semantic_projector.bias)
        else:
            self.semantic_projector = None

        # Initialize experts
        self.experts = nn.ModuleList([
            SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                dropout=config.dropout
            )
            for _ in range(config.n_routed_experts)
        ])

        self.shared_experts = nn.ModuleList([
            SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                dropout=config.dropout
            )
            for _ in range(config.n_shared_experts)
        ])

        # Null expert
        if config.use_null_expert:
            self.null_expert = SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                dropout=config.dropout
            )
        else:
            self.null_expert = None

        self.gate = TopkRouter(config)
        self.processor = ExpertProcessor(
            config,
            self.experts,
            self.shared_experts,
            self.null_expert
        )

    def forward(self, hidden_states: torch.Tensor, step: Optional[int] = None) -> ModelOutput:
        if step is not None:
            self.update_global_step(step)

        original_shape = hidden_states.shape
        moe_warmup_steps = getattr(self.config, "moe_warmup_steps", 0)

        if not self.enabled:
            shared_output = torch.stack([expert(hidden_states) for expert in self.shared_experts]).mean(dim=0)
            dummy_usage = torch.zeros(len(self.experts), device=hidden_states.device)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
            #self.log_debug("MoE disabled; only shared experts used")
            return ModelOutput(
                last_hidden_state=shared_output,
                aux_loss=aux_loss,
                expert_usage=dummy_usage,
                router_logits=None
            )

        if step is not None and step < moe_warmup_steps:
            hidden_states = hidden_states.view(-1, self.config.hidden_size)
            scale = step / moe_warmup_steps
            shared_output = torch.stack([expert(hidden_states) for expert in self.shared_experts]).mean(dim=0)
            expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts])
            routed_output = expert_outputs.mean(dim=0)
            output = (1 - scale) * shared_output + scale * routed_output
            expert_usage = torch.ones(len(self.experts), device=hidden_states.device) * hidden_states.size(0)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
            #self.log_debug(f"Warmup MoE step {step}: scale={scale:.2f}")
            return ModelOutput(
                last_hidden_state=output.view(original_shape),
                aux_loss=aux_loss,
                expert_usage=expert_usage,
                router_logits=None
            )

        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        topk_indices, topk_weights, aux_loss, router_logits = self.gate(hidden_states, step=step)
        output, expert_usage = self.processor.process_routed_experts(
            hidden_states, topk_indices, topk_weights, step=step
        )

        return ModelOutput(
            last_hidden_state=output.view(original_shape),
            aux_loss=aux_loss,
            expert_usage=expert_usage,
            router_logits=router_logits,
        )


    def calculate_imbalance(self, expert_counts: torch.Tensor):
        usage = expert_counts.float()
        total = usage.sum()
        if total < 1e-6:
            return torch.tensor(0.0)
        usage = usage / total
        imbalance = usage.max() / (usage.min() + 1e-6)
        return imbalance

class FlexibleFFN(nn.Module):
    """Flexible FFN with MoE that preserves base model compatibility"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.use_moe = config.use_moe

        # Always maintain standard FFN path
        self.ffn = SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        # Add MoE as parallel path
        if self.use_moe:
            self.moe = MoE(config)
            # Weighting factor between standard FFN and MoE
            self.moe_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_states: torch.Tensor, step: Optional[int] = None) -> ModelOutput:
        # Standard FFN output
        ffn_output = self.ffn(hidden_states)
        
        if self.use_moe:
            # MoE output
            moe_output = self.moe(hidden_states, step=step)
            moe_hidden = moe_output.last_hidden_state
            
            # Learnable combination
            alpha = torch.sigmoid(self.moe_alpha)
            combined = alpha * moe_hidden + (1 - alpha) * ffn_output
            
            return ModelOutput(
                last_hidden_state=combined,
                aux_loss=moe_output.aux_loss,
                expert_usage=moe_output.expert_usage,
                router_logits=moe_output.router_logits
            )
        else:
            return ModelOutput(
                last_hidden_state=ffn_output,
                aux_loss=torch.tensor(0.0, device=hidden_states.device),
                expert_usage=None,
                router_logits=None
            )

class DecoderLayer(nn.Module):
    """Transformer decoder layer with MoE integration"""
    
    def __init__(
        self,
        config: MoEModelConfig,
        layer_idx: int,
        alpha: float = 1.0,
        final_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.final_scale = final_scale
        self.alpha = alpha

        # Preserve base model structure
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Attention and FFN layers
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.ffn = FlexibleFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        step: Optional[int] = None,
    ) -> DecoderLayerOutput:
        residual = hidden_states
        
        # Normalize inputs
        attn_normed = self.input_layernorm(hidden_states)
        ffn_normed = self.post_attention_layernorm(hidden_states)
        
        # Compute attention
        attn_output, attn_weights = self.self_attn(
            hidden_states=attn_normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        
        # Compute FFN (standard or MoE)
        ffn_out = self.ffn(ffn_normed, step=step)
        ffn_output = ffn_out.last_hidden_state
        aux_loss = ffn_out.aux_loss
        
        # Combine paths
        combined = self.dropout(attn_output) + self.dropout(ffn_output)
        hidden_states = residual + self.alpha * combined
        hidden_states = hidden_states * self.final_scale
        
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weights=attn_weights,
            aux_loss=aux_loss,
            expert_usage=ffn_out.expert_usage,
        )

class TransformerModel(LlamaModel):
    """Modern Transformer model with caching and normalization support."""
    def __init__(self, config: MoEModelConfig) -> None:
        LlamaModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding and normalization layers
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using config settings."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)

    def _prepare_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        past_length: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create additive attention mask combining padding and causal masks."""
        total_len = past_length + seq_length
        
        # Create default mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, total_len), dtype=dtype, device=device)
        else:
            attention_mask = attention_mask.to(dtype=dtype, device=device)
            
            # Extend mask if needed
            if attention_mask.shape[1] < total_len:
                padding = torch.ones(
                    (batch_size, total_len - attention_mask.shape[1]),
                    dtype=dtype, device=device
                )
                attention_mask = torch.cat([attention_mask, padding], dim=-1)
        
        # Create combined padding and causal mask
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, total_len]
        causal_mask = torch.tril(torch.ones((total_len, total_len), device=device, dtype=dtype))
        causal_mask = causal_mask.view(1, 1, total_len, total_len)
        combined_mask = padding_mask * causal_mask
        
        # Convert to additive form
        additive_mask = (1.0 - combined_mask) * torch.finfo(dtype).min
        
        # Slice to current sequence
        if past_length > 0:
            additive_mask = additive_mask[:, :, -seq_length:, :]
        
        return additive_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_expert_usage: Optional[bool] = None,
        step: Optional[int] = None,
    ) -> ModelOutput: 
        """Transformer model forward pass with aux loss accumulation."""
        # Determine configuration
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        output_expert_usage = output_expert_usage or self.config.use_moe  

        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        # Prepare input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        past_length = 0

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=device
            ).expand(batch_size, seq_length)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if position_ids.max() >= self.config.max_position_embeddings:
            raise ValueError(
                f"Position id {position_ids.max()} exceeds max_position_embeddings {self.config.max_position_embeddings}."
            )

        # Attention mask
        additive_mask = self._prepare_attention_mask(
            batch_size=batch_size,
            seq_length=seq_length,
            past_length=past_length,
            attention_mask=attention_mask,
            device=device,
            dtype=dtype
        )

        # Outputs
        all_attentions = [] if output_attentions else None
        all_hidden_states = [] if output_hidden_states else None
        all_expert_usage = [] if output_expert_usage else None
        all_imbalances = [] if output_expert_usage else None
        aux_loss = torch.tensor(0.0, device=inputs_embeds.device)

        hidden_states = self.dropout(inputs_embeds)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # === accumulate auxiliary loss ===
        aux_loss = torch.tensor(0.0, device=inputs_embeds.device)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=additive_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                step=step,
            )

            hidden_states = layer_outputs.hidden_states

            # Aux loss is at index 2 if returned
            if layer_outputs.aux_loss is not None:
                aux_loss += layer_outputs.aux_loss

            # Attention
            if output_attentions:
                all_attentions.append(layer_outputs.attention_weights)

            # Expert usage
            if output_expert_usage and layer_outputs.expert_usage is not None:
                all_expert_usage.append(layer_outputs.expert_usage)
                imbalance = self.layers[i].ffn.moe.calculate_imbalance(layer_outputs.expert_usage)
                all_imbalances.append(imbalance)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            aux_loss=aux_loss,
            expert_usage=all_expert_usage,
            imbalances=all_imbalances
        )


class TransformerForCausalLM(TransformerModel, LlamaForCausalLM):
    """Transformer model with language modeling head for causal generation."""
    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__(config)
        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_expert_usage: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        step: Optional[int] = None,
    ) -> ModelOutput:
        """Forward pass with language modeling head and aux loss support."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_expert_usage=output_expert_usage,
            step=step,
        )
        
        # Compute language modeling head
        logits = self.lm_head(outputs.last_hidden_state)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = ce_loss + outputs.aux_loss

        return ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits=logits,
            loss=loss,
            aux_loss=outputs.aux_loss,
            expert_usage=outputs.expert_usage,
            imbalances=outputs.imbalances
        )

