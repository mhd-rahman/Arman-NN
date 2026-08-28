"""Training script for ArmanNN with pre-tokenized binary shard support.

Usage:
    # Train on pre-tokenized binary shards (recommended):
    python train.py --dataset_source binary --dataset_name ./data/pretrain/fineweb

    # Train on multiple binary shard directories (combined):
    python train.py --dataset_source binary --dataset_name ./data/pretrain/fineweb,./data/pretrain/code,./data/pretrain/wikipedia,./data/pretrain/math

    # Train on a HuggingFace dataset:
    python train.py --dataset_source huggingface --dataset_name wikitext --subset wikitext-2-raw-v1

    # Distributed training:
    torchrun --nproc_per_node=4 train.py --dataset_source binary --dataset_name ./data/pretrain/fineweb
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from arman.model import ArmanConfig, ArmanNN
from arman.training.data import load_dataset_from_source, BinaryShardDataset
from arman.training.scheduler import get_cosine_schedule_with_warmup
from arman.training.checkpointing import save_checkpoint, find_latest_checkpoint, load_checkpoint
from arman.training.distributed import setup_distributed, cleanup_distributed, is_main_process, get_data_sampler
from arman.training.evaluator import Evaluator, EvalConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed generation suite: prompts used at every eval checkpoint
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = [
    "The most important concept in machine learning is",
    "def fibonacci(n):\n",
    "The theory of relativity states that",
    "In mathematics, a prime number is",
    "The capital of France is",
    "To solve this equation, we first need to",
    "import torch\nimport torch.nn as nn\n\nclass Transformer(",
    "The quick brown fox",
]


# ---------------------------------------------------------------------------
# Downstream eval: simple completion accuracy on factual knowledge
# ---------------------------------------------------------------------------

DOWNSTREAM_TASKS = [
    # (prompt, expected_continuation_substring)
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("Water freezes at", "0"),
    ("The square root of 144 is", "12"),
    ("Python was created by", "Guido"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("HTML stands for", "Hypertext"),
    ("The speed of light is approximately", "300"),
    ("The chemical symbol for gold is", "Au"),
    ("The year World War II ended was", "1945"),
]


def run_generation_suite(model, tokenizer, device, max_new_tokens=50):
    """Run fixed generation prompts and return results."""
    from generate import generate

    model.eval()
    results = []

    for prompt in GENERATION_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": text})

    return results


def run_downstream_eval(model, tokenizer, device, max_new_tokens=20):
    """Run basic factual downstream eval and return accuracy."""
    from generate import generate

    model.eval()
    correct = 0

    for prompt, expected in DOWNSTREAM_TASKS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.01,  # near-greedy
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Check if expected substring appears in the continuation
        continuation = generated[len(prompt):]
        if expected.lower() in continuation.lower():
            correct += 1

    accuracy = correct / len(DOWNSTREAM_TASKS)
    return {"downstream_accuracy": accuracy, "downstream_correct": correct, "downstream_total": len(DOWNSTREAM_TASKS)}


def log_eval_table(step, tokens_seen, train_loss, eval_metrics, downstream, log_path):
    """Append a row to the CSV metrics log."""
    row = {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": f"{train_loss:.4f}",
        "val_loss": f"{eval_metrics.loss:.4f}",
        "val_ppl": f"{eval_metrics.perplexity:.2f}",
        "top1": f"{eval_metrics.top1_accuracy:.4f}",
        "mrr": f"{eval_metrics.mrr:.4f}",
        "downstream_acc": f"{downstream['downstream_accuracy']:.2f}",
    }

    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def build_eval_dataset(tokenizer, seq_len, cache_path="eval_dataset.pt"):
    """Load the pre-built eval dataset from disk.

    Expects a .pt file containing a list of (input_ids, targets) tuples.
    """
    cache = Path(cache_path)
    if not cache.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {cache}. "
            f"Place your eval_dataset.pt file in the working directory."
        )

    logger.info(f"Loading eval dataset from {cache}")
    samples = torch.load(cache, weights_only=False)
    logger.info(f"Eval dataset: {len(samples)} sequences")

    class ListDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    return ListDataset(samples)


def main():
    p = argparse.ArgumentParser(description="Train ArmanNN")

    # Dataset arguments
    p.add_argument("--dataset_source", type=str, default="binary", choices=["huggingface", "kaggle", "binary"],
                   help="Dataset source (default: binary)")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="Dataset path(s). For binary: comma-separated shard dirs. For HF: dataset name.")
    p.add_argument("--subset", type=str, default=None,
                   help="Dataset subset/config name (HuggingFace only)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="HuggingFace tokenizer name (default: gpt2)")
    p.add_argument("--max_samples", type=int, default=0)

    # Model arguments
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--mlp_hidden", type=int, default=2816)
    p.add_argument("--expert_hidden", type=int, default=1792)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--ssm_state_size", type=int, default=128)
    p.add_argument("--memory_slots", type=int, default=64)
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--use_graph", action="store_true", default=False)
    p.add_argument("--use_memory", action="store_true", default=False)

    # Training arguments
    p.add_argument("--steps", type=int, default=305000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # Checkpointing & eval
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--keep_checkpoints", type=int, default=5, help="Number of checkpoints to keep")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--metrics_log", type=str, default="training_metrics.csv",
                   help="CSV file for eval metrics log")
    p.add_argument("--eval_data", type=str, default=None,
                   help="Path to eval dataset: binary shard dir, .pt cache file, or 'none' to skip eval")

    # Misc
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

    # ================================================================
    # Load dataset
    # ================================================================
    if is_main_process():
        logger.info(f"Loading dataset: source={args.dataset_source}, name={args.dataset_name}")

    if args.dataset_source == "binary":
        # Support comma-separated dirs OR a parent dir with subfolders
        dirs = [d.strip() for d in args.dataset_name.split(",") if d.strip()]

        # If a single dir is given, check if it has subfolders with manifest.json
        if len(dirs) == 1:
            parent = Path(dirs[0])
            subfolders = sorted([
                d for d in parent.iterdir()
                if d.is_dir() and (d / "manifest.json").exists()
            ]) if parent.is_dir() and not (parent / "manifest.json").exists() else []

            if subfolders:
                # Parent dir with multiple shard subdirectories
                dirs = [str(d) for d in subfolders]
                if is_main_process():
                    logger.info(f"Found {len(dirs)} shard subdirectories in {parent}:")
            elif (parent / "manifest.json").exists():
                # Single shard dir
                dirs = [str(parent)]

        if len(dirs) == 1:
            dataset = BinaryShardDataset(dirs[0], seq_len=args.seq_len)
        else:
            datasets = [BinaryShardDataset(d, seq_len=args.seq_len) for d in dirs]
            dataset = ConcatDataset(datasets)
            if is_main_process():
                for d, ds in zip(dirs, datasets):
                    logger.info(f"  {Path(d).name}: {ds.total_tokens:,} tokens")

        # Get vocab size from first directory's manifest
        first_ds = dataset.datasets[0] if isinstance(dataset, ConcatDataset) else dataset
        vocab_size = first_ds.vocab_size
    else:
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
        from transformers import AutoTokenizer
        tokenizer_obj = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = tokenizer_obj.vocab_size

    if is_main_process():
        logger.info(f"Dataset ready: {len(dataset)} sequences of length {args.seq_len}")

    # ================================================================
    # Model
    # ================================================================
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
        use_graph=args.use_graph,
        use_memory=args.use_memory,
    )

    model = ArmanNN(cfg).to(device)
    if is_main_process():
        logger.info(f"ArmanNN | params={model.parameter_count():,} | vocab={vocab_size} | device={device}")

    # Wrap for distributed
    if world_size > 1:
        from arman.training.distributed import wrap_model_ddp
        model = wrap_model_ddp(model, local_rank)

    # ================================================================
    # Optimizer & scheduler
    # ================================================================
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

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps, min_lr_ratio=args.min_lr_ratio
    )

    # Mixed precision
    use_amp = args.use_amp and not args.no_amp
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ================================================================
    # Resume from checkpoint
    # ================================================================
    global_step = 0
    resume = args.resume and not args.no_resume
    if resume:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is not None:
            if is_main_process():
                logger.info(f"Resuming from {ckpt_path}")
            info = load_checkpoint(ckpt_path, model, optimizer=optimizer, scheduler=scheduler, device=device)
            global_step = info["step"]
            if is_main_process():
                logger.info(f"Resumed at step {global_step}")

    # ================================================================
    # Eval setup
    # ================================================================
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    eval_dataset = None
    if is_main_process():
        if args.eval_data and args.eval_data.lower() != "none":
            eval_path = Path(args.eval_data)
            if eval_path.suffix == ".pt":
                # Pre-cached .pt file
                eval_dataset = build_eval_dataset(tokenizer, args.seq_len, cache_path=str(eval_path))
            elif eval_path.is_dir():
                # Binary shard directory (single dir or parent with subfolders)
                subfolders = sorted([
                    d for d in eval_path.iterdir()
                    if d.is_dir() and (d / "manifest.json").exists()
                ]) if not (eval_path / "manifest.json").exists() else []

                if subfolders:
                    eval_datasets = [BinaryShardDataset(str(d), seq_len=args.seq_len) for d in subfolders]
                    eval_dataset = ConcatDataset(eval_datasets)
                    logger.info(f"Eval: {len(subfolders)} shard subdirs, {len(eval_dataset)} sequences")
                elif (eval_path / "manifest.json").exists():
                    eval_dataset = BinaryShardDataset(str(eval_path), seq_len=args.seq_len)
                    logger.info(f"Eval: {len(eval_dataset)} sequences from {eval_path}")
                else:
                    logger.warning(f"Eval path {eval_path} has no manifest.json or subfolders — skipping eval")
            else:
                logger.warning(f"Eval path {args.eval_data} not found — skipping eval")
        elif args.eval_data is None:
            # Fall back to eval_dataset.pt if it exists
            if Path("eval_dataset.pt").exists():
                eval_dataset = build_eval_dataset(tokenizer, args.seq_len)
            else:
                logger.info("No eval dataset specified and eval_dataset.pt not found — skipping eval")

        if eval_dataset is not None:
            logger.info(f"Eval dataset: {len(eval_dataset)} sequences")

    # ================================================================
    # DataLoader
    # ================================================================
    sampler = get_data_sampler(dataset, world_size, rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    tokens_per_step = args.batch_size * args.grad_accum * world_size * args.seq_len

    if is_main_process():
        logger.info(
            f"Training for {args.steps} steps | effective batch = "
            f"{args.batch_size * args.grad_accum * world_size} | "
            f"tokens/step = {tokens_per_step:,}"
        )

    # ================================================================
    # Training loop
    # ================================================================
    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    step_start = time.time()
    running_loss = 0.0
    running_count = 0

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
        running_loss += accum_loss
        running_count += 1

        # --------------------------------------------------------
        # Logging (every log_every steps)
        # --------------------------------------------------------
        if is_main_process() and global_step % args.log_every == 0:
            dt = time.time() - step_start
            lr = scheduler.get_last_lr()[0]
            tokens_seen = global_step * tokens_per_step
            tok_per_sec = args.log_every * tokens_per_step / dt
            logger.info(
                f"step={global_step:06d} | loss={accum_loss:.4f} | aux={accum_aux:.4f} | "
                f"grad_norm={grad_norm:.3f} | lr={lr:.2e} | "
                f"{tok_per_sec / 1e3:.1f}k tok/s | tokens={tokens_seen:,}"
            )
            step_start = time.time()

        # --------------------------------------------------------
        # Eval + Generation + Downstream (every eval_every steps)
        # --------------------------------------------------------
        if is_main_process() and global_step % args.eval_every == 0 and eval_dataset is not None:
            tokens_seen = global_step * tokens_per_step
            avg_train_loss = running_loss / running_count if running_count > 0 else 0.0

            # Validation metrics
            eval_cfg = EvalConfig(batch_size=16, use_amp=use_amp, amp_dtype="bfloat16")
            evaluator = Evaluator(model=model, eval_config=eval_cfg, device=device)
            eval_metrics = evaluator.evaluate(eval_dataset)

            # Downstream eval
            downstream = run_downstream_eval(model, tokenizer, device)

            # Log table row
            row = log_eval_table(
                step=global_step,
                tokens_seen=tokens_seen,
                train_loss=avg_train_loss,
                eval_metrics=eval_metrics,
                downstream=downstream,
                log_path=Path(args.metrics_log),
            )

            logger.info(
                f"\n{'='*70}\n"
                f"  EVAL @ step {global_step:,} | tokens seen: {tokens_seen:,}\n"
                f"  Train loss:     {avg_train_loss:.4f}\n"
                f"  Val loss:       {eval_metrics.loss:.4f}\n"
                f"  Val PPL:        {eval_metrics.perplexity:.2f}\n"
                f"  Top-1:          {eval_metrics.top1_accuracy:.4f}\n"
                f"  MRR:            {eval_metrics.mrr:.4f}\n"
                f"  Downstream:     {downstream['downstream_correct']}/{downstream['downstream_total']} "
                f"({downstream['downstream_accuracy']:.0%})\n"
                f"{'='*70}"
            )

            # Fixed generation suite
            gen_results = run_generation_suite(model, tokenizer, device)
            gen_log_path = Path(args.checkpoint_dir) / f"generations_step_{global_step:06d}.json"
            gen_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gen_log_path, "w") as f:
                json.dump({"step": global_step, "tokens_seen": tokens_seen, "generations": gen_results}, f, indent=2)
            logger.info(f"  Generations saved to {gen_log_path}")

            # Print a sample generation
            if gen_results:
                sample = gen_results[0]
                logger.info(f"  Sample: \"{sample['prompt']}\" → \"{sample['generated'][:100]}...\"")

            # Reset running loss tracker
            running_loss = 0.0
            running_count = 0
            model.train()

        # --------------------------------------------------------
        # Checkpoint (every save_every steps)
        # --------------------------------------------------------
        if is_main_process() and global_step % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            # Keep only last N checkpoints
            existing = sorted(Path(args.checkpoint_dir).glob("step_*.pt"))
            while len(existing) > args.keep_checkpoints:
                old = existing.pop(0)
                old.unlink()
                logger.info(f"Removed old checkpoint: {old}")

    # ================================================================
    # Final save
    # ================================================================
    if is_main_process():
        ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step:06d}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)
        logger.info(f"Training complete. Final checkpoint: {ckpt_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
