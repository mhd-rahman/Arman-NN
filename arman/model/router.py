import torch
from torch import nn


class PathRouter(nn.Module):
    def __init__(self, d_model: int, n_paths: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_paths, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.proj(x), dim=-1)
