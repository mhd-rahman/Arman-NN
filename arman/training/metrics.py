"""Token-level evaluation metrics: perplexity, top-k accuracy, MRR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class MetricsAccumulator:
    """Tracks running sums for computing aggregate metrics over a dataset.

    All values are accumulated on CPU to avoid holding GPU memory.
    """

    total_loss: float = 0.0
    total_tokens: int = 0
    total_correct_top1: int = 0
    total_correct_top5: int = 0
    total_correct_top10: int = 0
    total_mrr: float = 0.0  # Mean reciprocal rank
    n_batches: int = 0

    def update(
        self,
        loss: torch.Tensor | float,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> None:
        """Accumulate metrics from one batch.

        Args:
            loss: Scalar cross-entropy loss for this batch.
            logits: (batch, seq_len, vocab_size) raw model output.
            targets: (batch, seq_len) ground-truth token ids.
            ignore_index: Token id to ignore in accuracy computation.
        """
        with torch.no_grad():
            # Flatten
            logits_flat = logits.view(-1, logits.size(-1))  # (N, vocab)
            targets_flat = targets.view(-1)  # (N,)

            # Mask out ignored positions
            mask = targets_flat != ignore_index
            if mask.sum() == 0:
                return

            valid_logits = logits_flat[mask]
            valid_targets = targets_flat[mask]
            n_tokens = valid_targets.size(0)

            # Loss
            if isinstance(loss, torch.Tensor):
                self.total_loss += loss.item() * n_tokens
            else:
                self.total_loss += loss * n_tokens
            self.total_tokens += n_tokens

            # Top-k accuracy
            top10_preds = valid_logits.topk(min(10, valid_logits.size(-1)), dim=-1).indices
            correct = top10_preds == valid_targets.unsqueeze(-1)

            self.total_correct_top1 += correct[:, :1].any(dim=-1).sum().item()
            self.total_correct_top5 += correct[:, :5].any(dim=-1).sum().item()
            self.total_correct_top10 += correct.any(dim=-1).sum().item()

            # Mean Reciprocal Rank (MRR)
            # Find rank of correct token in top-10 (0-indexed), else rank=inf
            ranks = correct.float().argmax(dim=-1) + 1  # 1-indexed rank
            # Tokens not in top-10 get reciprocal rank of 0
            in_top10 = correct.any(dim=-1)
            reciprocal_ranks = torch.where(
                in_top10, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float)
            )
            self.total_mrr += reciprocal_ranks.sum().item()

            self.n_batches += 1

    def compute(self) -> EvalMetrics:
        """Compute final aggregate metrics."""
        if self.total_tokens == 0:
            return EvalMetrics()

        avg_loss = self.total_loss / self.total_tokens
        return EvalMetrics(
            loss=avg_loss,
            perplexity=math.exp(min(avg_loss, 100.0)),  # Clamp to avoid overflow
            top1_accuracy=self.total_correct_top1 / self.total_tokens,
            top5_accuracy=self.total_correct_top5 / self.total_tokens,
            top10_accuracy=self.total_correct_top10 / self.total_tokens,
            mrr=self.total_mrr / self.total_tokens,
            total_tokens=self.total_tokens,
            n_batches=self.n_batches,
        )

    def reset(self) -> None:
        """Reset all counters."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_correct_top10 = 0
        self.total_mrr = 0.0
        self.n_batches = 0


@dataclass
class EvalMetrics:
    """Container for computed evaluation metrics."""

    loss: float = 0.0
    perplexity: float = float("inf")
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    top10_accuracy: float = 0.0
    mrr: float = 0.0
    total_tokens: int = 0
    n_batches: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "top10_accuracy": self.top10_accuracy,
            "mrr": self.mrr,
            "total_tokens": self.total_tokens,
            "n_batches": self.n_batches,
        }

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} | ppl={self.perplexity:.2f} | "
            f"top1={self.top1_accuracy:.4f} | top5={self.top5_accuracy:.4f} | "
            f"top10={self.top10_accuracy:.4f} | mrr={self.mrr:.4f} | "
            f"tokens={self.total_tokens:,}"
        )
