import torch
from torch import nn


class GraphMessageLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.self_proj = nn.Linear(d_model, d_model, bias=False)
        self.neigh_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # nodes: [B, N, D], adjacency: [B, N, N]
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.bmm(adjacency, nodes) / degree
        return self.norm(nodes + torch.nn.functional.silu(self.self_proj(nodes) + self.neigh_proj(neigh)))


class GraphProcessor(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([GraphMessageLayer(d_model) for _ in range(n_layers)])
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, nodes: torch.Tensor | None, adjacency: torch.Tensor | None) -> torch.Tensor:
        if nodes is None or adjacency is None:
            return torch.zeros_like(x)
        h = nodes
        for layer in self.layers:
            h = layer(h, adjacency)
        q = self.query(x)
        k = self.key(h)
        v = self.value(h)
        scores = torch.einsum("btd,bnd->btn", q, k) / (x.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("btn,bnd->btd", weights, v)
