"""Full evaluation pipeline with distributed support."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import MetricsAccumulator, EvalMetrics
from .distributed import get_data_sampler, is_main_process

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation settings."""

    batch_size: int = 8
    num_workers: int = 2
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    max_batches: int = 0  # 0 = evaluate entire dataset


class Evaluator:
    """Runs model evaluation and computes comprehensive metrics.

    Supports distributed evaluation with metric reduction across ranks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eval_config: EvalConfig | None = None,
        device: torch.device | str = "cpu",
        world_size: int = 1,
        rank: int = 0,
    ):
        self.model = model
        self.cfg = eval_config or EvalConfig()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.world_size = world_size
        self.rank = rank
        self.amp_dtype = self._resolve_amp_dtype()

    def _resolve_amp_dtype(self) -> torch.dtype:
        if not self.cfg.use_amp:
            return torch.float32
        if self.cfg.amp_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.cfg.amp_dtype == "float16":
            return torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def _build_dataloader(self, dataset: Dataset) -> DataLoader:
        sampler = get_data_sampler(dataset, self.world_size, self.rank, shuffle=False)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> EvalMetrics:
        """Run evaluation over the full dataset.

        Args:
            dataset: Evaluation dataset. Items should be (input_ids, targets) tuples
                     or dicts with 'input_ids' and 'targets' keys.

        Returns:
            EvalMetrics with perplexity, top-k accuracy, MRR, etc.
        """
        self.model.eval()
        loader = self._build_dataloader(dataset)
        accumulator = MetricsAccumulator()

        for batch_idx, batch in enumerate(loader):
            if 0 < self.cfg.max_batches <= batch_idx:
                break

            # Unpack batch
            if isinstance(batch, (list, tuple)):
                input_ids, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

            # Forward pass
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.cfg.use_amp and self.device.type == "cuda"):
                out = self.model(input_ids, targets=targets)

            # Compute per-token loss (without aux) for perplexity
            logits = out["logits"]
            token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            accumulator.update(
                loss=token_loss,
                logits=logits,
                targets=targets,
                ignore_index=-100,
            )

        metrics = accumulator.compute()

        # Distributed reduction: average metrics across all ranks
        if self.world_size > 1:
            metrics = self._reduce_metrics(metrics)

        self.model.train()
        return metrics

    def _reduce_metrics(self, metrics: EvalMetrics) -> EvalMetrics:
        """Average metrics across distributed ranks."""
        import torch.distributed as dist

        tensor = torch.tensor(
            [
                metrics.loss,
                metrics.perplexity,
                metrics.top1_accuracy,
                metrics.top5_accuracy,
                metrics.top10_accuracy,
                metrics.mrr,
                float(metrics.total_tokens),
            ],
            device=self.device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size

        return EvalMetrics(
            loss=tensor[0].item(),
            perplexity=tensor[1].item(),
            top1_accuracy=tensor[2].item(),
            top5_accuracy=tensor[3].item(),
            top10_accuracy=tensor[4].item(),
            mrr=tensor[5].item(),
            total_tokens=int(tensor[6].item() * self.world_size),
            n_batches=metrics.n_batches,
        )
