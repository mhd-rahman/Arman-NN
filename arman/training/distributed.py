"""Distributed training setup utilities for DDP and FSDP."""

import os
import functools

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize the distributed process group.

    Reads RANK, WORLD_SIZE, LOCAL_RANK from environment (set by torchrun).

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_model_ddp(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> DDP:
    """Wrap a model in DistributedDataParallel.

    Args:
        model: Model already moved to the correct device.
        local_rank: Local GPU rank for this process.
        find_unused_parameters: Set True if some params aren't used every forward.

    Returns:
        DDP-wrapped model.
    """
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_model_fsdp(
    model: torch.nn.Module,
    mixed_precision: bool = True,
    cpu_offload: bool = False,
):
    """Wrap a model in Fully Sharded Data Parallel (FSDP).

    Args:
        model: Model (not yet moved to device — FSDP handles placement).
        mixed_precision: Whether to use bf16 mixed precision in FSDP.
        cpu_offload: Whether to offload parameters to CPU between forward/backward.

    Returns:
        FSDP-wrapped model.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, CPUOffload

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

    offload = CPUOffload(offload_params=True) if cpu_offload else None

    return FSDP(
        model,
        mixed_precision=mp_policy,
        cpu_offload=offload,
        use_orig_params=True,
    )


def get_data_sampler(dataset, world_size: int, rank: int, shuffle: bool = True):
    """Get a DistributedSampler if running distributed, else None.

    Args:
        dataset: The torch Dataset.
        world_size: Total number of processes.
        rank: Current process rank.
        shuffle: Whether to shuffle data.

    Returns:
        DistributedSampler or None.
    """
    if world_size <= 1:
        return None
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
