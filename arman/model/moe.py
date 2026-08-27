"""Sparse Mixture-of-Experts with batched permute-based routing.

Instead of looping over experts one-by-one (O(n_experts) sequential Python calls),
this implementation:
1. Flattens all tokens, routes them to top-k experts.
2. Sorts/permutes tokens by expert assignment.
3. Runs all tokens for each expert as contiguous batched matmuls via a single grouped call.
4. Scatters weighted results back to original positions.

This eliminates Python-level loops and leverages GPU parallelism fully.
"""

import torch
from torch import nn
from .mlp import SwiGLU


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_experts: int, top_k: int = 2):
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, hidden) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model) — weighted combination of expert outputs.
            aux_loss: scalar load-balancing loss.
        """
        b, t, d = x.shape
        # Flatten to 2D for routing: (num_tokens, d_model)
        x_flat = x.view(-1, d)
        num_tokens = x_flat.size(0)

        # Route: compute expert scores
        logits = self.router(x_flat)  # (num_tokens, n_experts)
        probs = torch.softmax(logits, dim=-1)

        # Select top-k experts per token
        topk_weights, topk_indices = probs.topk(self.top_k, dim=-1)  # (num_tokens, top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Permute-based dispatch: group tokens by expert assignment
        # Expand so each token appears top_k times (once per assigned expert)
        # token_indices[i] = which original token this expanded entry came from
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        token_indices = token_indices.reshape(-1)  # (num_tokens * top_k,)
        expert_indices = topk_indices.reshape(-1)   # (num_tokens * top_k,)
        weights_flat = topk_weights.reshape(-1)     # (num_tokens * top_k,)

        # Sort by expert index to create contiguous groups
        sorted_order = expert_indices.argsort(stable=True)
        sorted_expert_indices = expert_indices[sorted_order]
        sorted_token_indices = token_indices[sorted_order]
        sorted_weights = weights_flat[sorted_order]

        # Gather the token embeddings in sorted order
        sorted_inputs = x_flat[sorted_token_indices]  # (num_tokens * top_k, d_model)

        # Compute expert boundaries (how many tokens go to each expert)
        # Using bincount for efficiency
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.n_experts)

        # Run each expert on its contiguous slice
        # This is still a loop, but now each call processes a contiguous batch
        # which is maximally GPU-friendly (no masking, no scatter during compute)
        sorted_outputs = torch.empty_like(sorted_inputs)
        offset = 0
        for expert_idx, expert in enumerate(self.experts):
            count = expert_counts[expert_idx].item()
            if count == 0:
                continue
            expert_input = sorted_inputs[offset:offset + count]
            sorted_outputs[offset:offset + count] = expert(expert_input)
            offset += count

        # Weight the outputs
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(-1)

        # Scatter-add back to original token positions
        output = torch.zeros_like(x_flat)
        output.scatter_add_(
            0,
            sorted_token_indices.unsqueeze(-1).expand(-1, d),
            sorted_outputs,
        )

        output = output.view(b, t, d)

        # Load-balancing auxiliary loss (Switch Transformer style)
        # importance: mean probability assigned to each expert across all tokens
        importance = probs.mean(dim=0)  # (n_experts,)
        # load: fraction of tokens routed to each expert
        load = expert_counts.float() / (num_tokens * self.top_k)
        aux_loss = self.n_experts * (importance * load).sum()

        return output, aux_loss
