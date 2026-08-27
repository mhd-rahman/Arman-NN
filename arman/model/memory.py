import torch
from torch import nn


class NeuralMemory(nn.Module):
    """Differentiable memory bank with content-addressed reads and gated writes."""

    def __init__(self, d_model: int, slots: int, top_k: int):
        super().__init__()
        self.slots = slots
        self.top_k = top_k
        self.memory = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, 1)
        self.write_proj = nn.Linear(d_model, d_model, bias=False)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(self.query(x), dim=-1)
        m = torch.nn.functional.normalize(self.memory, dim=-1)
        scores = torch.einsum("btd,md->btm", q, m)
        k = min(self.top_k, self.slots)
        vals, idx = scores.topk(k, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        selected = self.memory[idx]
        return (selected * weights.unsqueeze(-1)).sum(dim=-2)

    def write_signal(self, x: torch.Tensor) -> torch.Tensor:
        # A differentiable candidate write used as an auxiliary recurrent signal.
        pooled = x.mean(dim=1)
        gate = torch.sigmoid(self.write_gate(pooled))
        return gate * self.write_proj(pooled)

    def forward(self, x: torch.Tensor):
        return self.read(x), self.write_signal(x)
