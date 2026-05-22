import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, token_ids: Int[Tensor, "..."]
    ) -> Float[Tensor, "... embedding_dim"]:
        return self.weight[token_ids]
