"""Stage-2 data preparation for ArmanNN — build the ~1.5B-token domain mix.

This is the "phase 2" counterpart to ``prepare/prepare_data.py``. It assembles a
curated ~1.5B token corpus from higher-quality knowledge/reasoning sources, plus a
replay slice drawn from the existing base-pretraining shards so a later stage-2
training run does not forget the stage-1 distribution.

Target mix (~1.5B tokens), each written to ``data/pretrained_2/<domain>/``:

    Wikipedia                   400M   wikimedia/wikipedia
    peS2o science               250M   allenai/peS2o
    FineWeb-Edu high-score      200M   HuggingFaceFW/fineweb-edu
    LibreTexts textbooks        100M   common-pile/libretexts_filtered
    DOAB books                   50M   common-pile/doab_filtered
    OpenWebMath                 150M   open-web-math/open-web-math
    StackExchange filtered       75M   common-pile/stackexchange_filtered
    USGPO / reference            75M   common-pile/usgpo
    Base-pretraining replay     225M   combined mixture of ALL data/pretrained/* folders (15%)

Usage:
    # Prepare the entire stage-2 mix:
    python prepare/prepare_data_2.py

    # Re-prepare everything even if manifests already exist:
    python prepare/prepare_data_2.py --overwrite

    # Prepare only specific domains:
    python prepare/prepare_data_2.py --only wikipedia openwebmath base_replay

    # Smaller shards so the total lands closer to 1.5B (recommended):
    python prepare/prepare_data_2.py --shard_size 25000000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Make the sibling prepare_data module importable (this script lives in prepare/),
# and put the repo root on the path too for consistency with the rest of the repo.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from prepare_data import prepare_dataset, save_shard, _save_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------

STAGE2_ROOT = Path("../../data/pretrained_2")    # all stage-2 shards land here
BASE_DATA_ROOT = Path("../../data/pretrained")   # stage-1 shards used for replay


# ---------------------------------------------------------------------------
# Stage-2 data mix (~1.5B tokens)
# ---------------------------------------------------------------------------
# Each entry becomes a subfolder under data/pretrained_2/<name> with binary shards.
# "base_replay" is special: it is not a HuggingFace dataset — it is resampled from
# the combined mixture of all existing data/pretrained/* folders.

DATA_SOURCES = [
    {
        "name": "wikipedia",
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "text_column": "text",
        "target_tokens": 400_000_000,
    },
    {
        "name": "science_pes2o",
        "dataset": "allenai/peS2o",
        "subset": None,
        "text_column": "text",
        "target_tokens": 250_000_000,
    },
    {
        "name": "fineweb_edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_column": "text",
        "target_tokens": 200_000_000,
    },
    {
        "name": "libretexts",
        "dataset": "common-pile/libretexts_filtered",
        "subset": None,
        "text_column": "text",
        "target_tokens": 100_000_000,
    },
    {
        "name": "doab_books",
        "dataset": "common-pile/doab_filtered",
        "subset": None,
        "text_column": "text",
        "target_tokens": 50_000_000,
    },
    {
        "name": "openwebmath",
        "dataset": "open-web-math/open-web-math",
        "subset": None,
        "text_column": "text",
        "target_tokens": 150_000_000,
    },
    {
        "name": "stackexchange",
        "dataset": "common-pile/stackexchange_filtered",
        "subset": None,
        "text_column": "text",
        "target_tokens": 75_000_000,
    },
    {
        "name": "usgpo",
        "dataset": "common-pile/usgpo",
        "subset": None,
        "text_column": "text",
        "target_tokens": 75_000_000,
    },
    {
        # Base-pretraining replay (15%): resampled from the COMBINED mixture of all
        # data/pretrained/* folders (not a single folder). Handled specially.
        "name": "base_replay",
        "dataset": None,
        "subset": None,
        "text_column": None,
        "target_tokens": 225_000_000,
    },
]

TARGET_TOTAL_TOKENS = sum(s["target_tokens"] for s in DATA_SOURCES)  # 1.5B


# ---------------------------------------------------------------------------
# Base-pretraining replay: resample from the COMBINED mixture of ALL
# data/pretrained/* folders and write fresh shards under data/pretrained_2/base_replay.
# ---------------------------------------------------------------------------

def prepare_base_replay(
    output_dir: Path,
    base_root: Path = BASE_DATA_ROOT,
    target_tokens: int = 225_000_000,
    shard_size: int = 100_000_000,
    seed: int = 42,
    max_seq_len: int = 1024,
) -> dict:
    """Build a replay slice by sampling proportionally across every base source folder.

    Rather than copying a single base folder, this reads all data/pretrained/* shard
    dirs, samples contiguous chunks from each in proportion to its size, interleaves
    them, and writes new binary shards. This preserves the stage-1 mixture.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover base source folders (each must have a manifest.json)
    source_dirs = sorted(
        d for d in base_root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ) if base_root.is_dir() else []

    if not source_dirs:
        raise FileNotFoundError(
            f"No base shard folders with manifest.json found under {base_root}. "
            f"Base replay requires stage-1 shards to exist first."
        )

    logger.info(f"Base replay: found {len(source_dirs)} base source folders under {base_root}")

    # Load each source's manifest + memory-map its shards.
    sources = []
    total_available = 0
    dtype = None
    vocab_size = None
    eos_id = None
    tokenizer_name = None

    for source_dir in source_dirs:
        with open(source_dir / "manifest.json") as f:
            manifest = json.load(f)
        src_dtype = np.dtype(manifest["format"]["dtype"])
        if dtype is None:
            dtype = src_dtype
            vocab_size = manifest["tokenizer"]["vocab_size"]
            eos_id = manifest["tokenizer"]["eos_token_id"]
            tokenizer_name = manifest["tokenizer"]["name"]
        elif src_dtype != dtype:
            raise ValueError(
                f"Base source {source_dir.name} uses dtype {src_dtype} but expected {dtype}. "
                f"All base shards must share a tokenizer/dtype for replay."
            )

        mmaps = []
        offsets = []
        offset = 0
        for shard_meta in manifest["shards"]:
            shard_path = source_dir / shard_meta["filename"]
            if shard_path.exists():
                mmap = np.memmap(shard_path, dtype=src_dtype, mode="r")
                mmaps.append(mmap)
                offsets.append(offset)
                offset += len(mmap)

        if offset == 0:
            logger.warning(f"  {source_dir.name}: no readable shards, skipping")
            continue

        sources.append({"name": source_dir.name, "mmaps": mmaps, "offsets": offsets, "tokens": offset})
        total_available += offset
        logger.info(f"  {source_dir.name:<16} {offset:>14,} tokens")

    if total_available == 0:
        raise RuntimeError("Base replay: base folders contained no readable tokens.")

    target_tokens = min(target_tokens, total_available)
    logger.info(
        f"Base replay: sampling {target_tokens:,} tokens from {total_available:,} available "
        f"(combined mixture of {len(sources)} sources)"
    )

    rng = random.Random(seed)

    def read_range(mmaps, offsets, start, end):
        """Read a contiguous token range across shard boundaries within one source."""
        # Find starting shard
        shard_idx = 0
        for i, off in enumerate(offsets):
            if off <= start:
                shard_idx = i
            else:
                break
        result = []
        remaining = end - start
        pos = start
        while remaining > 0 and shard_idx < len(mmaps):
            shard_start = offsets[shard_idx]
            shard = mmaps[shard_idx]
            local_start = pos - shard_start
            available = len(shard) - local_start
            take = min(available, remaining)
            result.append(np.asarray(shard[local_start:local_start + take]))
            remaining -= take
            pos += take
            shard_idx += 1
        return np.concatenate(result) if len(result) > 1 else result[0]

    # Proportional per-source token budget, drawn as random contiguous chunks
    # so we get variety without loading whole sources into RAM.
    CHUNK = max_seq_len * 64  # ~64 sequences per sampled chunk

    progress_path = output_dir / "progress.json"
    shard_id = 0
    tokens_written = 0
    shards_meta = []
    buffer: list[int] = []

    for src in sources:
        src["remaining"] = int(round(target_tokens * src["tokens"] / total_available))

    active = [s for s in sources if s["remaining"] > 0]
    t_start = time.time()

    while active and tokens_written < target_tokens:
        rng.shuffle(active)
        for src in list(active):
            if tokens_written >= target_tokens:
                break
            take = min(CHUNK, src["remaining"], target_tokens - tokens_written)
            if take <= 0:
                src["remaining"] = 0
                active.remove(src)
                continue
            max_start = max(0, src["tokens"] - take)
            start = rng.randint(0, max_start)
            chunk = read_range(src["mmaps"], src["offsets"], start, start + take)
            buffer.extend(int(t) for t in chunk.tolist())
            src["remaining"] -= take

            while len(buffer) >= shard_size:
                shard_tokens = buffer[:shard_size]
                buffer = buffer[shard_size:]
                meta = save_shard(shard_tokens, shard_id, output_dir, dtype)
                shards_meta.append(meta)
                tokens_written += len(shard_tokens)
                shard_id += 1
                _save_progress(progress_path, shard_id, tokens_written, 0, shards_meta)

            if src["remaining"] <= 0 and src in active:
                active.remove(src)
        active = [s for s in active if s["remaining"] > 0]

    # Flush remaining buffer
    if buffer:
        meta = save_shard(buffer, shard_id, output_dir, dtype)
        shards_meta.append(meta)
        tokens_written += len(buffer)
        shard_id += 1

    elapsed = time.time() - t_start
    manifest = {
        "version": "pretrain-v1",
        "dataset": {
            "source": "base_replay",
            "subset": None,
            "split": "train",
            "text_column": None,
            "replay_sources": [s["name"] for s in sources],
            "replay_root": str(base_root),
        },
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
        },
        "format": {
            "dtype": np.dtype(dtype).name,
            "shard_size": shard_size,
            "sequence_length": max_seq_len,
        },
        "stats": {
            "total_tokens": tokens_written,
            "total_shards": len(shards_meta),
            "documents_processed": 0,
            "preparation_time_seconds": round(elapsed, 1),
        },
        "shards": shards_meta,
        "seed": seed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if progress_path.exists():
        progress_path.unlink()

    logger.info(
        f"Base replay done: {tokens_written:,} tokens in {len(shards_meta)} shards "
        f"(combined from {len(sources)} base sources) | {elapsed:.1f}s"
    )
    return manifest


# ---------------------------------------------------------------------------
# Build the full stage-2 data mix
# ---------------------------------------------------------------------------

def prepare_stage2_data(args) -> None:
    STAGE2_ROOT.mkdir(parents=True, exist_ok=True)

    selected = DATA_SOURCES
    if args.only:
        wanted = set(args.only)
        selected = [s for s in DATA_SOURCES if s["name"] in wanted]
        unknown = wanted - {s["name"] for s in DATA_SOURCES}
        if unknown:
            logger.warning(f"Ignoring unknown domain(s): {sorted(unknown)}")
        if not selected:
            raise SystemExit(
                f"No matching domains for --only {args.only}. "
                f"Valid: {[s['name'] for s in DATA_SOURCES]}"
            )

    selected_total = sum(s["target_tokens"] for s in selected)
    logger.info(
        f"Preparing {len(selected)}/{len(DATA_SOURCES)} stage-2 domains "
        f"(~{selected_total:,} target tokens) into {STAGE2_ROOT}"
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    prepared = []
    for source in selected:
        out_dir = STAGE2_ROOT / source["name"]
        manifest_path = out_dir / "manifest.json"

        if manifest_path.exists() and not args.overwrite:
            logger.info(f"[SKIP] {source['name']}: manifest already exists at {out_dir}")
            prepared.append((source["name"], _read_total_tokens(manifest_path)))
            continue

        logger.info(f"\n{'='*64}\nPreparing: {source['name']} ({source['target_tokens']:,} tokens)\n{'='*64}")

        if source["name"] == "base_replay":
            manifest = prepare_base_replay(
                output_dir=out_dir,
                base_root=BASE_DATA_ROOT,
                target_tokens=source["target_tokens"],
                shard_size=args.shard_size,
                seed=args.seed,
                max_seq_len=args.seq_len,
            )
        else:
            manifest = prepare_dataset(
                dataset_name=source["dataset"],
                output_dir=out_dir,
                tokenizer_name=args.tokenizer,
                target_tokens=source["target_tokens"],
                shard_size=args.shard_size,
                text_column=source["text_column"],
                split="train",
                subset=source["subset"],
                streaming=not args.no_streaming,
                seed=args.seed,
                max_seq_len=args.seq_len,
                hf_token=hf_token,
            )
        prepared.append((source["name"], manifest["stats"]["total_tokens"]))

    # Summary
    grand_total = sum(t for _, t in prepared)
    logger.info(f"\n{'='*64}\nStage-2 preparation summary\n{'='*64}")
    for name, tokens in prepared:
        logger.info(f"  {name:<16} {tokens:>14,} tokens")
    logger.info(f"  {'TOTAL':<16} {grand_total:>14,} tokens")
    logger.info(f"Shards written under {STAGE2_ROOT}")


def _read_total_tokens(manifest_path: Path) -> int:
    try:
        with open(manifest_path) as f:
            return json.load(f)["stats"]["total_tokens"]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Prepare the ArmanNN stage-2 (~1.5B token) data mix into data/pretrained_2/."
    )
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="HuggingFace tokenizer name (default: gpt2)")
    p.add_argument("--shard_size", type=int, default=100_000_000,
                   help="Tokens per shard (default: 100M). Use a smaller value (e.g. 25M) "
                        "to land closer to the exact 1.5B target.")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="Sequence length (metadata only, default: 1024)")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle/sample seed (default: 42)")
    p.add_argument("--no_streaming", action="store_true",
                   help="Disable streaming (loads full dataset into memory)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-prepare domains even if a manifest already exists")
    p.add_argument("--only", nargs="+", default=None,
                   help="Prepare only these domain names (e.g. --only wikipedia base_replay)")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace token for gated datasets (or set HF_TOKEN env var)")
    return p


def main():
    args = build_parser().parse_args()
    prepare_stage2_data(args)


if __name__ == "__main__":
    main()
