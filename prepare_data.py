"""Pre-tokenize datasets from HuggingFace and write binary shards for fast training.

This script implements:
    HuggingFace streaming download → tokenize → pack into fixed-length shards → write .bin files → manifest.json

Usage:
    # Prepare FineWeb-Edu (7.5B tokens):
    python prepare_data.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --target_tokens 7_500_000_000 \
        --output_dir ./data/pretrain/fineweb \
        --tokenizer gpt2

    # Prepare code data (1.5B tokens):
    python prepare_data.py \
        --dataset bigcode/starcoderdata \
        --target_tokens 1_500_000_000 \
        --output_dir ./data/pretrain/code \
        --tokenizer gpt2 \
        --text_column content

    # Prepare Wikipedia (1B tokens):
    python prepare_data.py \
        --dataset wikimedia/wikipedia \
        --subset 20231101.en \
        --target_tokens 1_000_000_000 \
        --output_dir ./data/pretrain/wikipedia \
        --tokenizer gpt2

    # Prepare OpenWebMath (2B tokens):
    python prepare_data.py \
        --dataset open-web-math/open-web-math \
        --target_tokens 2_000_000_000 \
        --output_dir ./data/pretrain/math \
        --tokenizer gpt2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def save_shard(tokens: list[int], shard_id: int, output_dir: Path, dtype: np.dtype) -> dict:
    """Write a shard of token IDs to a binary file and return metadata."""
    arr = np.array(tokens, dtype=dtype)
    filename = f"train-{shard_id:05d}.bin"
    path = output_dir / filename
    arr.tofile(path)

    # Compute checksum for reproducibility
    sha256 = hashlib.sha256(arr.tobytes()).hexdigest()

    meta = {
        "filename": filename,
        "tokens": len(arr),
        "bytes": arr.nbytes,
        "sha256": sha256,
    }
    logger.info(
        f"  Shard {shard_id:05d}: {len(arr):>12,} tokens | "
        f"{arr.nbytes / 1e6:.1f} MB | {sha256[:12]}..."
    )
    return meta


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset_name: str,
    output_dir: Path,
    tokenizer_name: str = "gpt2",
    target_tokens: int = 1_000_000_000,
    shard_size: int = 100_000_000,
    text_column: str = "text",
    split: str = "train",
    subset: str | None = None,
    streaming: bool = True,
    seed: int = 42,
    max_seq_len: int = 1024,
    hf_token: str | None = None,
) -> dict:
    """Download, tokenize, and shard a dataset.

    Args:
        dataset_name: HuggingFace dataset identifier.
        output_dir: Directory to write binary shards into.
        tokenizer_name: HuggingFace tokenizer name/path.
        target_tokens: Stop after this many tokens (approximate).
        shard_size: Number of tokens per shard file.
        text_column: Column name containing text.
        split: Dataset split to use.
        subset: Dataset config/subset name.
        streaming: Use streaming mode (recommended for large datasets).
        seed: Shuffle seed for streaming datasets.
        max_seq_len: Sequence length (for manifest metadata only).
        hf_token: HuggingFace access token for gated datasets (e.g. StarCoderData).

    Returns:
        Manifest dict describing the prepared data.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    # Determine dtype: uint16 if vocab < 65536, else uint32
    if vocab_size < 65536:
        dtype = np.uint16
        dtype_str = "uint16"
    else:
        dtype = np.uint32
        dtype_str = "uint32"

    logger.info(f"Vocab size: {vocab_size} → using {dtype_str}")
    logger.info(f"Target: {target_tokens:,} tokens in shards of {shard_size:,}")

    # Load dataset (streaming for large datasets)
    logger.info(f"Loading dataset: {dataset_name} (subset={subset}, split={split}, streaming={streaming})")
    load_kwargs = {"split": split, "streaming": streaming}
    if hf_token:
        load_kwargs["token"] = hf_token
    if subset:
        ds = load_dataset(dataset_name, subset, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    # Tokenize and write shards
    buffer: list[int] = []
    shard_id = 0
    tokens_written = 0
    docs_processed = 0
    shards_meta: list[dict] = []
    t_start = time.time()

    logger.info("Starting tokenization...")

    for example in ds:
        text = example.get(text_column, "")
        if not text or not text.strip():
            continue

        # Tokenize with EOS separator between documents
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)

        buffer.extend(ids)
        docs_processed += 1

        # Flush full shards
        while len(buffer) >= shard_size:
            shard_tokens = buffer[:shard_size]
            buffer = buffer[shard_size:]

            meta = save_shard(shard_tokens, shard_id, output_dir, dtype)
            shards_meta.append(meta)
            tokens_written += len(shard_tokens)
            shard_id += 1

            # Progress logging
            if shard_id % 5 == 0:
                elapsed = time.time() - t_start
                rate = tokens_written / elapsed
                eta = (target_tokens - tokens_written) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {tokens_written:,}/{target_tokens:,} tokens "
                    f"({100 * tokens_written / target_tokens:.1f}%) | "
                    f"{rate / 1e6:.1f}M tok/s | ETA: {eta / 60:.0f} min"
                )

        # Stop at target token count
        if tokens_written >= target_tokens:
            break

    # Write remaining buffer as final (potentially smaller) shard
    if buffer and tokens_written < target_tokens:
        meta = save_shard(buffer, shard_id, output_dir, dtype)
        shards_meta.append(meta)
        tokens_written += len(buffer)

    elapsed = time.time() - t_start
    logger.info(
        f"Done! {tokens_written:,} tokens in {shard_id + 1} shards | "
        f"{docs_processed:,} documents | {elapsed:.1f}s"
    )

    # Build manifest
    manifest = {
        "version": "pretrain-v1",
        "dataset": {
            "source": dataset_name,
            "subset": subset,
            "split": split,
            "text_column": text_column,
        },
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
        },
        "format": {
            "dtype": dtype_str,
            "shard_size": shard_size,
            "sequence_length": max_seq_len,
        },
        "stats": {
            "total_tokens": tokens_written,
            "total_shards": len(shards_meta),
            "documents_processed": docs_processed,
            "preparation_time_seconds": round(elapsed, 1),
        },
        "shards": shards_meta,
        "seed": seed,
    }

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_shards(output_dir: Path) -> bool:
    """Validate shard integrity against manifest checksums."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    dtype = np.dtype(manifest["format"]["dtype"])
    errors = 0

    for shard_meta in manifest["shards"]:
        path = output_dir / shard_meta["filename"]
        if not path.exists():
            logger.error(f"Missing shard: {path}")
            errors += 1
            continue

        arr = np.fromfile(path, dtype=dtype)
        if len(arr) != shard_meta["tokens"]:
            logger.error(
                f"Token count mismatch in {path}: "
                f"expected {shard_meta['tokens']}, got {len(arr)}"
            )
            errors += 1
            continue

        sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
        if sha256 != shard_meta["sha256"]:
            logger.error(f"Checksum mismatch in {path}")
            errors += 1
        else:
            logger.info(f"  ✓ {shard_meta['filename']}")

    if errors == 0:
        logger.info("All shards validated successfully.")
    else:
        logger.error(f"Validation failed: {errors} error(s)")

    return errors == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Pre-tokenize HuggingFace datasets into binary shards for ArmanNN training."
    )

    p.add_argument("--dataset", type=str, required=True,
                   help="HuggingFace dataset identifier (e.g. 'HuggingFaceFW/fineweb-edu')")
    p.add_argument("--subset", type=str, default=None,
                   help="Dataset subset/config (e.g. 'sample-10BT')")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split (default: train)")
    p.add_argument("--text_column", type=str, default="text",
                   help="Text column name (default: text)")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="HuggingFace tokenizer name (default: gpt2)")
    p.add_argument("--target_tokens", type=int, default=1_000_000_000,
                   help="Target number of tokens to extract (default: 1B)")
    p.add_argument("--shard_size", type=int, default=100_000_000,
                   help="Tokens per shard (default: 100M ≈ 200MB with uint16)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for binary shards")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="Sequence length (metadata only, default: 1024)")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for streaming (default: 42)")
    p.add_argument("--no_streaming", action="store_true",
                   help="Disable streaming (loads full dataset into memory)")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace access token for gated datasets (or set HF_TOKEN env var)")
    p.add_argument("--validate", action="store_true",
                   help="Validate existing shards against manifest (skip preparation)")

    args = p.parse_args()
    output_dir = Path(args.output_dir)

    if args.validate:
        success = validate_shards(output_dir)
        raise SystemExit(0 if success else 1)

    # Resolve HF token: CLI arg > env var
    import os as _os
    hf_token = args.hf_token or _os.environ.get("HF_TOKEN")

    prepare_dataset(
        dataset_name=args.dataset,
        output_dir=output_dir,
        tokenizer_name=args.tokenizer,
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        text_column=args.text_column,
        split=args.split,
        subset=args.subset,
        streaming=not args.no_streaming,
        seed=args.seed,
        max_seq_len=args.seq_len,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
