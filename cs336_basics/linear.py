import torch
from beartype import beartype
from einops import einsum
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "... in_features"]
    ) -> Float[Tensor, "... out_features"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    @jaxtyped(typechecker=beartype)
    def _forward(
        self, x: Float[Tensor, "... in_features"]
    ) -> Float[Tensor, "... out_features"]:
        # Torch-native equivalent of forward(), kept for reference.
        # Either of these matches the einops version above:
        #     return torch.einsum("...i,oi->...o", x, self.weight)
        #     return x @ self.weight.T
        return x @ self.weight.T
