"""Build the eval_dataset.pt file for ArmanNN training evaluation.

Combines:
  1. Wikitext-2 test set (full) — fixed external benchmark
  2. 7500 random sequences sampled from pre-tokenized training shards (in-domain mix)

Usage:
    python build_eval_dataset.py --data_dir ./data/pretrain --output eval_dataset.pt

    # Custom in-domain sample count:
    python build_eval_dataset.py --data_dir ./data/pretrain --output eval_dataset.pt --in_domain_samples 10000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def _read_range(mmaps, offsets, start, end):
    """Read a contiguous range of tokens across shard boundaries."""
    # Binary search for starting shard
    shard_idx = 0
    for i, off in enumerate(offsets):
        if off <= start:
            shard_idx = i
        else:
            break

    result = []
    remaining = end - start
    pos = start

    while remaining > 0:
        shard_start = offsets[shard_idx]
        shard = mmaps[shard_idx]
        local_start = pos - shard_start
        available = len(shard) - local_start
        take = min(available, remaining)
        result.append(shard[local_start:local_start + take])
        remaining -= take
        pos += take
        shard_idx += 1

    return np.concatenate(result) if len(result) > 1 else result[0]


def main():
    p = argparse.ArgumentParser(description="Build eval_dataset.pt for ArmanNN")
    p.add_argument("--data_dir", type=str, default="./data/pretrain",
                   help="Root directory containing binary shard subdirectories")
    p.add_argument("--output", type=str, default="eval_dataset.pt",
                   help="Output path for the eval dataset (default: eval_dataset.pt)")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="Sequence length (default: 1024)")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="Tokenizer for Wikitext-2 (default: gpt2)")
    p.add_argument("--in_domain_samples", type=int, default=7500,
                   help="Number of in-domain sequences to sample from training shards (default: 7500)")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed for in-domain sampling (default: 12345)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    all_samples = []

    # ==================================================================
    # 1. Wikitext-2 test set (full fixed benchmark)
    # ==================================================================
    print("Loading Wikitext-2 test set...")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eval_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    wikitext_tokens = []
    for example in eval_ds:
        text = example["text"]
        if text.strip():
            wikitext_tokens.extend(tokenizer.encode(text, add_special_tokens=False))

    wikitext_samples = []
    for i in range(0, len(wikitext_tokens) - args.seq_len, args.seq_len):
        x = torch.tensor(wikitext_tokens[i:i + args.seq_len], dtype=torch.long)
        y = torch.tensor(wikitext_tokens[i + 1:i + args.seq_len + 1], dtype=torch.long)
        wikitext_samples.append((x, y))

    print(f"  Wikitext-2 test: {len(wikitext_samples)} sequences")
    all_samples.extend(wikitext_samples)
    del wikitext_tokens, eval_ds

    # ==================================================================
    # 2. In-domain samples from all training shard sources
    # ==================================================================
    print(f"\nSampling {args.in_domain_samples} in-domain sequences from training shards...")

    # Read shards directly with numpy — no arman package needed
    all_tokens = []
    for source_dir in sorted(data_dir.iterdir()):
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        dtype = np.dtype(manifest["format"]["dtype"])
        source_tokens = 0
        for shard_meta in manifest["shards"]:
            shard_path = source_dir / shard_meta["filename"]
            if shard_path.exists():
                source_tokens += shard_meta["tokens"]
        all_tokens.append((source_dir, manifest, dtype, source_tokens))
        print(f"  {source_dir.name:<12} {source_tokens:>14,} tokens")

    if not all_tokens:
        print(f"WARNING: No shard directories found in {data_dir}. Only Wikitext-2 will be in eval set.")
    else:
        # Calculate total sequences available across all sources
        total_seqs = sum(
            (tokens - 1) // args.seq_len for _, _, _, tokens in all_tokens
        )
        print(f"  Total available: {total_seqs:,} sequences")

        n_samples = min(args.in_domain_samples, total_seqs)
        random.seed(args.seed)

        # Sample proportionally from each source
        in_domain_samples = []
        for source_dir, manifest, dtype, source_tokens in all_tokens:
            n_seqs = (source_tokens - 1) // args.seq_len
            # Proportional allocation
            source_n = max(1, int(n_samples * n_seqs / total_seqs))
            source_indices = random.sample(range(n_seqs), min(source_n, n_seqs))

            # Memory-map all shards for this source
            mmaps = []
            offsets = []
            offset = 0
            for shard_meta in manifest["shards"]:
                shard_path = source_dir / shard_meta["filename"]
                if shard_path.exists():
                    mmap = np.memmap(shard_path, dtype=dtype, mode="r")
                    mmaps.append(mmap)
                    offsets.append(offset)
                    offset += len(mmap)

            # Extract sequences at sampled indices
            for idx in source_indices:
                start = idx * args.seq_len
                end = start + args.seq_len + 1
                # Read tokens across shard boundaries
                tokens_buf = _read_range(mmaps, offsets, start, end)
                x = torch.from_numpy(tokens_buf[:args.seq_len].astype(np.int64))
                y = torch.from_numpy(tokens_buf[1:args.seq_len + 1].astype(np.int64))
                in_domain_samples.append((x, y))

            print(f"    {source_dir.name}: sampled {len(source_indices)} sequences")

        all_samples.extend(in_domain_samples)
        print(f"  In-domain total: {len(in_domain_samples)} sequences")
        del in_domain_samples

    # ==================================================================
    # 3. Save
    # ==================================================================
    output_path = Path(args.output)
    torch.save(all_samples, output_path)

    print(f"\n{'='*60}")
    print(f"Eval dataset saved to: {output_path}")
    print(f"Total: {len(all_samples)} sequences")
    print(f"  - Wikitext-2 test: {len(wikitext_samples)}")
    print(f"  - In-domain mix:   {len(all_samples) - len(wikitext_samples)}")
    print(f"  - File size:       {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
