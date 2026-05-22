import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from .linear import Linear


@jaxtyped(typechecker=beartype)
def silu(x: Float[Tensor, " ... d_ff"]) -> Float[Tensor, " ... d_ff"]:
    """Apply the SiLU activation function to a tensor.

    Args:
        x (Float[Tensor, " ... d_ff"]): The input tensor.

    Returns:
        Float[Tensor, " ... d_ff"]: The output tensor.
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1x: Float[Tensor, " ... d_ff"] = self.linear1(x)
        w3x: Float[Tensor, " ... d_ff"] = self.linear3(x)
        return self.linear2(silu(w1x) * w3x)
