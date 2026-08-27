"""Production training loop with mixed precision, gradient accumulation,
distributed training, LR scheduling, and resumable checkpointing."""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from arman.model import ArmanConfig, ArmanNN
from .scheduler import get_cosine_schedule_with_warmup
from .checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_ddp,
    wrap_model_fsdp,
    get_data_sampler,
)
from .evaluator import Evaluator, EvalConfig
from .metrics import EvalMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters and settings."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Schedule
    warmup_steps: int = 100
    total_steps: int = 10000
    min_lr_ratio: float = 0.1

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # Distributed
    parallel_mode: str = "ddp"  # "ddp", "fsdp", or "none"
    backend: str = "nccl"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500
    resume: bool = True  # Auto-resume from latest checkpoint

    # Logging
    log_every_steps: int = 10
    eval_every_steps: int = 500

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"


class Trainer:
    """Full-featured training loop for ArmanNN."""

    def __init__(
        self,
        model_config: ArmanConfig,
        train_config: TrainConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        self.model_config = model_config
        self.cfg = train_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Distributed setup
        self.rank, self.local_rank, self.world_size = setup_distributed(self.cfg.backend)
        self.device = self._resolve_device()

        # Model
        self.model = ArmanNN(model_config).to(self.device)
        if is_main_process():
            logger.info(f"Model parameters: {self.model.parameter_count():,}")

        # Wrap for distributed
        self.model = self._wrap_model(self.model)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.cfg.total_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

        # Mixed precision
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.cfg.use_amp and self.amp_dtype == torch.float16))

        # State
        self.global_step = 0

        # Resume from checkpoint
        if self.cfg.resume:
            self._try_resume()

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.local_rank}")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def _resolve_amp_dtype(self) -> torch.dtype:
        if not self.cfg.use_amp:
            return torch.float32
        if self.cfg.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.world_size <= 1 or self.cfg.parallel_mode == "none":
            return model
        if self.cfg.parallel_mode == "fsdp":
            return wrap_model_fsdp(model, mixed_precision=self.cfg.use_amp)
        return wrap_model_ddp(model, self.local_rank)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # Separate weight decay for non-bias, non-norm params
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            param_groups, lr=self.cfg.learning_rate, betas=(self.cfg.beta1, self.cfg.beta2)
        )

    def _try_resume(self) -> None:
        ckpt_path = find_latest_checkpoint(self.cfg.checkpoint_dir)
        if ckpt_path is None:
            if is_main_process():
                logger.info("No checkpoint found, starting from scratch.")
            return
        if is_main_process():
            logger.info(f"Resuming from {ckpt_path}")
        info = load_checkpoint(
            ckpt_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.global_step = info["step"]
        if is_main_process():
            logger.info(f"Resumed at step {self.global_step}")

    def _build_dataloader(self) -> DataLoader:
        sampler = get_data_sampler(self.train_dataset, self.world_size, self.rank)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True,
        )

    def _train_step(self, batch: dict | tuple) -> dict[str, float]:
        """Single gradient accumulation micro-step."""
        if isinstance(batch, (list, tuple)):
            input_ids, targets = batch[0].to(self.device), batch[1].to(self.device)
            kwargs = {"input_ids": input_ids, "targets": targets}
        else:
            kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.cfg.use_amp):
            out = self.model(**kwargs)
            loss = out["loss"] / self.cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        return {
            "loss": out["loss"].detach().item(),
            "aux_loss": out["aux_loss"].detach().item() if out.get("aux_loss") is not None else 0.0,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on eval_dataset with full metrics (perplexity, top-k accuracy, MRR)."""
        if self.eval_dataset is None:
            return {}

        eval_config = EvalConfig(
            batch_size=self.cfg.batch_size,
            use_amp=self.cfg.use_amp,
            amp_dtype=self.cfg.amp_dtype,
        )
        evaluator = Evaluator(
            model=self.model,
            eval_config=eval_config,
            device=self.device,
            world_size=self.world_size,
            rank=self.rank,
        )
        metrics = evaluator.evaluate(self.eval_dataset)

        if is_main_process():
            logger.info(
                f"[eval] step={self.global_step:06d} | {metrics}"
            )

        return metrics.to_dict()

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        dataloader = self._build_dataloader()
        data_iter = iter(dataloader)

        if is_main_process():
            logger.info(
                f"Training for {self.cfg.total_steps} steps | "
                f"batch_size={self.cfg.batch_size} x grad_accum={self.cfg.gradient_accumulation_steps} "
                f"x world_size={self.world_size} = "
                f"{self.cfg.batch_size * self.cfg.gradient_accumulation_steps * self.world_size} effective"
            )

        self.optimizer.zero_grad(set_to_none=True)
        step_start = time.time()

        while self.global_step < self.cfg.total_steps:
            metrics = {"loss": 0.0, "aux_loss": 0.0}

            for _micro in range(self.cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                step_metrics = self._train_step(batch)
                metrics["loss"] += step_metrics["loss"] / self.cfg.gradient_accumulation_steps
                metrics["aux_loss"] += step_metrics["aux_loss"] / self.cfg.gradient_accumulation_steps

            # Unscale, clip, step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            self.global_step += 1

            # Logging
            if is_main_process() and self.global_step % self.cfg.log_every_steps == 0:
                elapsed = time.time() - step_start
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"step={self.global_step:06d} | "
                    f"loss={metrics['loss']:.4f} | aux={metrics['aux_loss']:.4f} | "
                    f"grad_norm={grad_norm:.3f} | lr={lr:.2e} | "
                    f"dt={elapsed:.2f}s"
                )
                step_start = time.time()

            # Checkpointing
            if is_main_process() and self.global_step % self.cfg.save_every_steps == 0:
                ckpt_path = Path(self.cfg.checkpoint_dir) / f"step_{self.global_step:06d}.pt"
                save_checkpoint(
                    ckpt_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.global_step,
                    self.model_config,
                    extra={"loss": metrics["loss"]},
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # Evaluation
            if self.eval_dataset is not None and self.global_step % self.cfg.eval_every_steps == 0:
                self.evaluate()

        # Final save
        if is_main_process():
            ckpt_path = Path(self.cfg.checkpoint_dir) / f"step_{self.global_step:06d}.pt"
            save_checkpoint(
                ckpt_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.global_step,
                self.model_config,
            )
            logger.info(f"Training complete. Final checkpoint: {ckpt_path}")

        cleanup_distributed()
