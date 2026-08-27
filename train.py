"""Training script for ArmanNN with HuggingFace and Kaggle dataset support.

Usage:
    # Train on a HuggingFace dataset:
    python train.py --dataset_source huggingface --dataset_name wikitext --subset wikitext-2-raw-v1

    # Train on a Kaggle dataset:
    python train.py --dataset_source kaggle --dataset_name username/dataset-name

    # With custom tokenizer and sequence length:
    python train.py --dataset_source huggingface --dataset_name openwebtext \\
        --tokenizer gpt2 --seq_len 512 --steps 5000

    # Distributed training:
    torchrun --nproc_per_node=4 train.py --dataset_source huggingface --dataset_name openwebtext
"""

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from arman.model import ArmanConfig, ArmanNN
from arman.training.data import load_dataset_from_source
from arman.training.scheduler import get_cosine_schedule_with_warmup
from arman.training.checkpointing import save_checkpoint, find_latest_checkpoint, load_checkpoint
from arman.training.distributed import setup_distributed, cleanup_distributed, is_main_process, get_data_sampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Train ArmanNN")

    # Dataset arguments
    p.add_argument("--dataset_source", type=str, required=True, choices=["huggingface", "kaggle"],
                   help="Dataset source: 'huggingface' or 'kaggle'")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="Dataset identifier (e.g. 'wikitext' for HF, 'username/dataset' for Kaggle)")
    p.add_argument("--subset", type=str, default=None,
                   help="Dataset subset/config name (HuggingFace only, e.g. 'wikitext-2-raw-v1')")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split (default: train)")
    p.add_argument("--text_column", type=str, default="text",
                   help="Name of the text column in the dataset (default: text)")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="HuggingFace tokenizer name (default: gpt2)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Max raw samples to load (0 = all)")

    # Model arguments
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--mlp_hidden", type=int, default=1024)
    p.add_argument("--expert_hidden", type=int, default=1024)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--ssm_state_size", type=int, default=64)
    p.add_argument("--memory_slots", type=int, default=64)
    p.add_argument("--graph_layers", type=int, default=2)

    # Training arguments
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Auto-resume from latest checkpoint")
    p.add_argument("--no_resume", action="store_true")

    # Misc
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Distributed setup
    rank, local_rank, world_size = setup_distributed(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )

    # Device
    if args.device == "auto":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load dataset
    if is_main_process():
        logger.info(f"Loading dataset: source={args.dataset_source}, name={args.dataset_name}")

    dataset = load_dataset_from_source(
        source=args.dataset_source,
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        subset=args.subset,
    )

    if is_main_process():
        logger.info(f"Dataset ready: {len(dataset)} sequences of length {args.seq_len}")

    # Get vocab size from tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size

    # Model config
    cfg = ArmanConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        mlp_hidden=args.mlp_hidden,
        expert_hidden=args.expert_hidden,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        ssm_state_size=args.ssm_state_size,
        memory_slots=args.memory_slots,
        graph_layers=args.graph_layers,
    )

    model = ArmanNN(cfg).to(device)
    if is_main_process():
        logger.info(f"ArmanNN | params={model.parameter_count():,} | vocab={vocab_size} | device={device}")

    # Wrap for distributed
    if world_size > 1:
        from arman.training.distributed import wrap_model_ddp
        model = wrap_model_ddp(model, local_rank)

    # Optimizer (weight decay separation)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps, min_lr_ratio=args.min_lr_ratio
    )

    # Mixed precision
    use_amp = args.use_amp and not args.no_amp
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # Resume from checkpoint
    global_step = 0
    resume = args.resume and not args.no_resume
    if resume:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is not None:
            if is_main_process():
                logger.info(f"Resuming from {ckpt_path}")
            info = load_checkpoint(ckpt_path, model, optimizer=optimizer, scheduler=scheduler, device=device)
            global_step = info["step"]

    # DataLoader
    sampler = get_data_sampler(dataset, world_size, rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process():
        effective_batch = args.batch_size * args.grad_accum * world_size
        logger.info(f"Training for {args.steps} steps | effective batch size={effective_batch}")

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.steps:
        accum_loss = 0.0
        accum_aux = 0.0

        for _micro in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x, targets=y)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += out["loss"].item() / args.grad_accum
            accum_aux += out["aux_loss"].item() / args.grad_accum

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1

        if is_main_process() and global_step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"step={global_step:06d} | loss={accum_loss:.4f} | "
                f"aux={accum_aux:.4f} | grad_norm={grad_norm:.3f} | lr={lr:.2e}"
            )

        if is_main_process() and global_step % args.save_every == 0:
            from pathlib import Path
            ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step:06d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    if is_main_process():
        from pathlib import Path
        ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step:06d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)
        logger.info(f"Training complete. Final checkpoint: {ckpt_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
