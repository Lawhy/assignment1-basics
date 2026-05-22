import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[Tensor, " ... 1"] = (einsum(x, x, "... d_model, ... d_model -> ...").unsqueeze(-1) / self.d_model + self.eps) ** 0.5
        normed: Float[Tensor, " ... d_model"] = x / rms
        result: Float[Tensor, " ... d_model"] = normed * self.weight
        return result.to(in_dtype)
