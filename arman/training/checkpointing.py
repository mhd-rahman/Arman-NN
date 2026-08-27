"""Resumable checkpoint save/load utilities."""

import os
from pathlib import Path
from dataclasses import asdict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    step: int,
    config: object,
    extra: dict | None = None,
) -> None:
    """Save a full training checkpoint for resumption.

    Args:
        path: File path to save the checkpoint.
        model: The model (handles DDP-wrapped models automatically).
        optimizer: The optimizer state.
        scheduler: LR scheduler state (optional).
        step: Current global training step.
        config: ArmanConfig dataclass instance.
        extra: Any additional state to persist (e.g. best_loss, epoch).
    """
    # Unwrap DDP/FSDP if needed
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra is not None:
        state["extra"] = extra

    # Write to temp file then rename for atomicity
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.rename(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: Checkpoint file path.
        model: The model to load weights into (unwrapped).
        optimizer: Optimizer to restore state into (optional).
        scheduler: Scheduler to restore state into (optional).
        device: Device to map tensors to during load.

    Returns:
        Dict with 'step', 'config', and optional 'extra' keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device, weights_only=False)

    # Handle DDP-wrapped model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return {
        "step": state["step"],
        "config": state["config"],
        "extra": state.get("extra", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint file in a directory (by step number).

    Expects filenames like 'step_000100.pt' or 'checkpoint_100.pt'.
    Falls back to most recently modified file if naming convention doesn't match.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pt_files = list(checkpoint_dir.glob("*.pt"))
    if not pt_files:
        return None

    # Try to extract step numbers from filenames
    def _extract_step(p: Path) -> int:
        import re
        match = re.search(r"(\d+)", p.stem)
        return int(match.group(1)) if match else 0

    pt_files.sort(key=_extract_step)
    return pt_files[-1]
