import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k  # per-head dimension d_k = d_model / n_heads
        self.max_seq_len = max_seq_len
        k_range = torch.arange(1, d_k // 2 + 1, device=device).float()
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = 1 / (theta ** ((2 * k_range - 2) / d_k))
        angles = einsum(positions, freqs, "p, k -> p k")
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos: Float[Tensor, " ... seq_len d_k_half"] = self.cos_cache[token_positions]
        sin: Float[Tensor, " ... seq_len d_k_half"] = self.sin_cache[token_positions]
        x_even: Float[Tensor, " ... seq_len d_k_half"] = x[..., 0::2]
        x_odd: Float[Tensor, " ... seq_len d_k_half"] = x[..., 1::2]
        rotated_even: Float[Tensor, " ... seq_len d_k_half"] = x_even * cos - x_odd * sin
        rotated_odd: Float[Tensor, " ... seq_len d_k_half"] = x_even * sin + x_odd * cos
        # Interleave rotated_even and rotated_odd back into adjacent-pair layout:
        # stack adds an inner size-2 dim, flatten merges (d_k/2, 2) → d_k.
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2)
