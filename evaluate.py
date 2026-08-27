"""Standalone evaluation script for ArmanNN.

Usage:
    python evaluate.py --checkpoint checkpoints/step_001000.pt --dataset toy --batch_size 16

For distributed evaluation:
    torchrun --nproc_per_node=4 evaluate.py --checkpoint ... --dataset toy
"""

import argparse
import logging

import torch
from torch.utils.data import Dataset

from arman.model import ArmanConfig, ArmanNN
from arman.training.evaluator import Evaluator, EvalConfig
from arman.training.distributed import setup_distributed, cleanup_distributed, is_main_process
from arman.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ToyEvalDataset(Dataset):
    """Deterministic evaluation dataset — shift-by-1 pattern for sanity checking."""

    def __init__(self, vocab_size: int = 256, seq_len: int = 64, samples: int = 500, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.samples = samples
        self.seed = seed
        # Pre-generate for determinism
        gen = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, vocab_size, (samples,), generator=gen)
        self.data = torch.stack([
            (torch.arange(seq_len) + s) % vocab_size for s in starts
        ])

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x = self.data[idx].long()
        y = (x + 1) % self.vocab_size
        return x, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate ArmanNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy"], help="Evaluation dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max_batches", type=int, default=0, help="Max batches to evaluate (0 = all)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocab size for toy dataset")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for toy dataset")
    parser.add_argument("--samples", type=int, default=500, help="Number of eval samples")
    args = parser.parse_args()

    # Distributed setup
    rank, local_rank, world_size = setup_distributed(backend="nccl" if torch.cuda.is_available() else "gloo")

    # Device
    if args.device == "auto":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    if is_main_process():
        logger.info(f"Loading checkpoint: {args.checkpoint}")

    # Load config from checkpoint
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ArmanConfig(**state["config"])
    model = ArmanNN(config).to(device)
    model.load_state_dict(state["model"])

    if is_main_process():
        logger.info(f"Model loaded | params={model.parameter_count():,} | device={device}")

    # Dataset
    if args.dataset == "toy":
        eval_dataset = ToyEvalDataset(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            samples=args.samples,
        )

    # Evaluate
    eval_config = EvalConfig(
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        max_batches=args.max_batches,
    )
    evaluator = Evaluator(
        model=model,
        eval_config=eval_config,
        device=device,
        world_size=world_size,
        rank=rank,
    )

    if is_main_process():
        logger.info(f"Evaluating on {len(eval_dataset)} samples...")

    metrics = evaluator.evaluate(eval_dataset)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Loss:           {metrics.loss:.4f}")
        logger.info(f"  Perplexity:     {metrics.perplexity:.2f}")
        logger.info(f"  Top-1 Accuracy: {metrics.top1_accuracy:.4f} ({metrics.top1_accuracy*100:.2f}%)")
        logger.info(f"  Top-5 Accuracy: {metrics.top5_accuracy:.4f} ({metrics.top5_accuracy*100:.2f}%)")
        logger.info(f"  Top-10 Accuracy:{metrics.top10_accuracy:.4f} ({metrics.top10_accuracy*100:.2f}%)")
        logger.info(f"  MRR:            {metrics.mrr:.4f}")
        logger.info(f"  Total Tokens:   {metrics.total_tokens:,}")
        logger.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
