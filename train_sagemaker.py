"""
ArmanNN SageMaker Training Script

Production-ready training script for AWS SageMaker with:
- Multi-GPU distributed training (DDP via torchrun)
- Automatic checkpoint resume (spot instance friendly)
- SageMaker environment path conventions
- Hyperparameter parsing from SM environment
- Mixed domain streaming data (FineWeb-Edu, Code, Wikipedia, Math)
- Wikitext-2 held-out evaluation
- bf16 mixed precision + gradient checkpointing

Usage (local testing):
    python train_sagemaker.py --total_steps 100 --batch_size 8

Usage (SageMaker — handled by launch.py):
    Automatically invoked via torchrun by SageMaker's PyTorch estimator.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.environ.get("SM_MODULE_DIR", "."))
sys.path.insert(0, "/opt/ml/code")
sys.path.insert(0, ".")

from arman.model import ArmanConfig, ArmanNN
from arman.training.scheduler import get_cosine_schedule_with_warmup
from arman.training.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from arman.training.evaluator import Evaluator, EvalConfig
from arman.training.distributed import setup_distributed, cleanup_distributed, is_main_process

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("arman.sagemaker")


# ============================================================
# SAGEMAKER ENVIRONMENT
# ============================================================
def get_sagemaker_paths():
    """Get SageMaker standard paths, falling back to local defaults."""
    return {
        "model_dir": os.environ.get("SM_MODEL_DIR", "./model_output"),
        "checkpoint_dir": os.environ.get("SM_CHECKPOINT_DIR", 
                          os.environ.get("SM_OUTPUT_DATA_DIR", "./checkpoints")),
        "output_dir": os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        "num_gpus": int(os.environ.get("SM_NUM_GPUS", torch.cuda.device_count())),
    }


# ============================================================
# DATA PIPELINE (same as notebook)
# ============================================================
def stream_hf_dataset(dataset_name, subset=None, split="train", text_field="text",
                      data_dir=None, token=None, seed=42, skip=0):
    kwargs = {"path": dataset_name, "split": split, "streaming": True, "token": token}
    if subset is not None:
        kwargs["name"] = subset
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    ds = load_dataset(**kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if skip > 0:
        ds = ds.skip(skip)
    for example in ds:
        text = example.get(text_field)
        if text and isinstance(text, str) and text.strip():
            yield text


def fineweb_stream(seed=42, skip=0, token=None):
    yield from stream_hf_dataset("HuggingFaceFW/fineweb-edu", subset="sample-10BT",
                                  split="train", seed=seed, skip=skip, token=token)


def wikipedia_stream(seed=42, skip=0, token=None):
    yield from stream_hf_dataset("wikimedia/wikipedia", subset="20231101.en",
                                  split="train", seed=seed, skip=skip, token=token)


def math_stream(seed=42, skip=0, token=None):
    yield from stream_hf_dataset("open-web-math/open-web-math", split="train",
                                  seed=seed, skip=skip, token=token)


CODE_LANGUAGES = {
    "python": 0.267, "javascript": 0.120, "typescript": 0.100,
    "cpp": 0.087, "java": 0.087, "c": 0.073, "rust": 0.067,
    "go": 0.067, "shell": 0.033, "sql": 0.033, "html": 0.020,
    "css": 0.013, "json": 0.013, "yaml": 0.010, "markdown": 0.010,
}
total_code_weight = sum(CODE_LANGUAGES.values())
CODE_LANGUAGES = {k: v / total_code_weight for k, v in CODE_LANGUAGES.items()}


def code_stream(seed=42, skip=0, token=None):
    languages = list(CODE_LANGUAGES.keys())
    weights = [CODE_LANGUAGES[l] for l in languages]
    streams = {
        lang: iter(stream_hf_dataset("bigcode/starcoderdata", split="train",
                                      text_field="content", data_dir=lang,
                                      token=token, seed=seed + i, skip=skip))
        for i, lang in enumerate(languages)
    }
    rng = random.Random(seed)
    while True:
        lang = rng.choices(languages, weights=weights, k=1)[0]
        try:
            yield next(streams[lang])
        except StopIteration:
            streams[lang] = iter(stream_hf_dataset(
                "bigcode/starcoderdata", split="train", text_field="content",
                data_dir=lang, token=token, seed=seed + 1000 + languages.index(lang), skip=0))


DATASET_WEIGHTS = {"fineweb": 0.625, "code": 0.125, "wikipedia": 0.083, "math": 0.167}


class MixedPretrainingDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, seed=42, validation=False, token=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.validation = validation
        self.token = token

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        seed = self.seed + worker_id
        skip = 50_000 if self.validation else 0
        if self.validation:
            seed += 10_000

        streams = {
            "fineweb": iter(fineweb_stream(seed=seed + 1, skip=skip, token=self.token)),
            "code": iter(code_stream(seed=seed + 2, skip=skip, token=self.token)),
            "wikipedia": iter(wikipedia_stream(seed=seed + 3, skip=skip, token=self.token)),
            "math": iter(math_stream(seed=seed + 4, skip=skip, token=self.token)),
        }
        names = list(DATASET_WEIGHTS.keys())
        weights = [DATASET_WEIGHTS[n] for n in names]
        rng = random.Random(seed)
        token_buffer = []

        while True:
            source = rng.choices(names, weights=weights, k=1)[0]
            try:
                text = next(streams[source])
            except StopIteration:
                rebuild_seed = seed + {"fineweb": 100, "code": 200, "wikipedia": 300, "math": 400}[source]
                if source == "fineweb":
                    streams[source] = iter(fineweb_stream(seed=rebuild_seed, token=self.token))
                elif source == "code":
                    streams[source] = iter(code_stream(seed=rebuild_seed, token=self.token))
                elif source == "wikipedia":
                    streams[source] = iter(wikipedia_stream(seed=rebuild_seed, token=self.token))
                elif source == "math":
                    streams[source] = iter(math_stream(seed=rebuild_seed, token=self.token))
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            if self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_eval_dataset(tokenizer, seq_len):
    """Build held-out eval from Wikitext-2 + randomized mixed domain."""
    # Wikitext-2 test (fixed benchmark)
    logger.info("Loading Wikitext-2 test for evaluation...")
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wikitext_tokens = []
    for example in eval_ds:
        text = example["text"]
        if text.strip():
            wikitext_tokens.extend(tokenizer.encode(text, add_special_tokens=False))

    wikitext_samples = []
    for i in range(0, len(wikitext_tokens) - seq_len, seq_len):
        x = torch.tensor(wikitext_tokens[i:i + seq_len], dtype=torch.long)
        y = torch.tensor(wikitext_tokens[i + 1:i + seq_len + 1], dtype=torch.long)
        wikitext_samples.append((x, y))

    # Mixed domain eval (randomized)
    logger.info("Loading mixed domain eval sequences...")
    mixed_eval_stream = MixedPretrainingDataset(
        tokenizer=tokenizer, seq_len=seq_len,
        seed=random.randint(0, 100000), validation=True,
    )
    mixed_samples = []
    for i, (x, y) in enumerate(mixed_eval_stream):
        mixed_samples.append((x, y))
        if i >= 2499:
            break

    all_samples = wikitext_samples + mixed_samples
    logger.info(f"Eval dataset: {len(wikitext_samples)} wikitext + {len(mixed_samples)} mixed = {len(all_samples)} total")
    return ListDataset(all_samples)


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="ArmanNN SageMaker Training")

    # Model
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlp_hidden", type=int, default=2816)
    parser.add_argument("--expert_hidden", type=int, default=1792)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--ssm_state_size", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50257)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=76000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Checkpointing & logging
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)

    # Infrastructure
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None)

    # SageMaker passes hyperparameters as command-line args
    # Also check SM_HPS environment variable
    args = parser.parse_args()

    # Override from SM_HPS if present (SageMaker JSON hyperparameters)
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        try:
            hps = json.loads(sm_hps)
            for key, value in hps.items():
                if hasattr(args, key):
                    arg_type = type(getattr(args, key))
                    setattr(args, key, arg_type(value))
        except (json.JSONDecodeError, ValueError):
            pass

    return args


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def train(args):
    # Seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Paths
    paths = get_sagemaker_paths()
    checkpoint_dir = paths["checkpoint_dir"]
    model_dir = paths["model_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Distributed setup
    rank, local_rank, world_size = setup_distributed(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        logger.info(f"SageMaker Training | world_size={world_size} | device={device}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        logger.info(f"Model output dir: {model_dir}")
        logger.info(f"Hyperparameters: {vars(args)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Model
    config = ArmanConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        mlp_hidden=args.mlp_hidden,
        expert_hidden=args.expert_hidden,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        ssm_state_size=args.ssm_state_size,
        use_graph=False,
        use_memory=False,
    )

    model = ArmanNN(config).to(device)

    if args.gradient_checkpointing and not args.no_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main_process():
            logger.info("Gradient checkpointing: enabled")

    if is_main_process():
        logger.info(f"Model parameters: {model.parameter_count():,}")

    # DDP wrapping for multi-GPU
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    # Optimizer
    raw_model = model.module if hasattr(model, "module") else model
    decay_params = [p for n, p in raw_model.named_parameters()
                    if p.requires_grad and p.ndim >= 2 and "norm" not in n]
    no_decay_params = [p for n, p in raw_model.named_parameters()
                       if p.requires_grad and (p.ndim < 2 or "norm" in n)]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95))

    # Resume from checkpoint (spot instance recovery)
    global_step = 0
    ckpt = find_latest_checkpoint(checkpoint_dir)
    if ckpt:
        if is_main_process():
            logger.info(f"Resuming from checkpoint: {ckpt}")
        info = load_checkpoint(ckpt, model, optimizer=optimizer, scheduler=None,
                               device="cpu", strict=False)
        global_step = info["step"]
        if is_main_process():
            logger.info(f"Resumed at step {global_step}")

    # Scheduler (created after resume so we can fast-forward)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio
    )
    for _ in range(global_step):
        scheduler.step()

    # Data
    train_dataset = MixedPretrainingDataset(
        tokenizer=tokenizer, seq_len=args.max_seq_len,
        seed=args.seed, validation=False, token=args.hf_token,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Eval dataset (main process only to avoid redundant downloads)
    eval_dataset = None
    if is_main_process():
        eval_dataset = build_eval_dataset(tokenizer, args.max_seq_len)

    # Training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(train_loader)
    step_start = time.time()
    AMP_DTYPE = torch.bfloat16

    if is_main_process():
        effective_batch = args.batch_size * args.grad_accum * world_size
        logger.info(
            f"Training for {args.total_steps} steps | "
            f"effective batch = {effective_batch} | "
            f"tokens/step = {effective_batch * args.max_seq_len:,}"
        )
        logger.info(f"Starting from step {global_step}")

    while global_step < args.total_steps:
        accum_loss = 0.0
        accum_aux = 0.0

        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                out = model(x, targets=y)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            accum_loss += out["loss"].item() / args.grad_accum
            accum_aux += out["aux_loss"].item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        # Logging
        if is_main_process() and global_step % args.log_every == 0:
            dt = time.time() - step_start
            lr = scheduler.get_last_lr()[0]
            tokens_seen = global_step * args.batch_size * args.grad_accum * world_size * args.max_seq_len
            logger.info(
                f"step={global_step:06d} | loss={accum_loss:.4f} | "
                f"aux={accum_aux:.4f} | grad_norm={grad_norm:.3f} | "
                f"lr={lr:.2e} | dt={dt:.2f}s | tokens={tokens_seen:,}"
            )
            step_start = time.time()

        # Checkpoint (all ranks wait, but only rank 0 saves)
        if global_step % args.save_every == 0:
            if world_size > 1:
                dist.barrier()
            if is_main_process():
                ckpt_path = Path(checkpoint_dir) / f"step_{global_step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, config)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # Evaluation (rank 0 only)
        if is_main_process() and global_step % args.eval_every == 0 and eval_dataset is not None:
            eval_cfg = EvalConfig(batch_size=16, use_amp=True, amp_dtype="bfloat16")
            evaluator = Evaluator(model=model, eval_config=eval_cfg, device=device)
            metrics = evaluator.evaluate(eval_dataset)
            logger.info(f"[eval] step={global_step:06d} | {metrics}")
            model.train()

    # Final save
    if world_size > 1:
        dist.barrier()
    if is_main_process():
        # Save to checkpoint dir
        ckpt_path = Path(checkpoint_dir) / f"step_{global_step:06d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, config)

        # Also save to model_dir for SageMaker model artifact
        final_model_path = Path(model_dir) / "arman_nn_final.pt"
        save_checkpoint(final_model_path, model, optimizer, scheduler, global_step, config)
        logger.info(f"Training complete! Model saved to {final_model_path}")

        # Save config as JSON for easy loading
        config_path = Path(model_dir) / "config.json"
        import dataclasses
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)
        logger.info(f"Config saved to {config_path}")

    cleanup_distributed()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)
