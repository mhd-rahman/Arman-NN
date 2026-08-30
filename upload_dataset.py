"""Upload a local data directory (with subfolders) to a HuggingFace dataset repo.

Mirrors the auth/upload style used in export.py: HfApi + create_repo + folder
upload, token from --hf_token or the HF_TOKEN env var (falling back to cached
login via `huggingface-cli login`).

The directory structure is preserved as-is in the repo. For example:

    data/
      train/
        shard_0.parquet
        shard_1.parquet
      val/
        shard_0.parquet

...uploads to the repo with the same train/ and val/ layout.

Usage:
    # Upload a data folder to an existing dataset repo, passing the token directly:
    python upload_dataset.py \\
        --hf_token hf_xxx \\
        --repo_id your-username/YourDataset \\
        --data_dir /path/to/data

    # Token can also come from the HF_TOKEN env var or a cached `huggingface-cli login`:
    python upload_dataset.py --repo_id your-username/YourDataset --data_dir /path/to/data

    # Private repo (only affects repo creation if it does not exist yet):
    python upload_dataset.py --hf_token hf_xxx --repo_id your-username/YourDataset \\
        --data_dir /path/to/data --private

    # Only upload a subset by pattern (e.g. just parquet files):
    python upload_dataset.py --repo_id your-username/YourDataset --data_dir /path/to/data \\
        --allow_patterns "*.parquet"

    # Upload only specific subfolders:
    python upload_dataset.py --repo_id your-username/YourDataset --data_dir /path/to/data \\
        --allow_patterns "train/*" "val/*"

    # Dry run: list what would be uploaded without pushing:
    python upload_dataset.py --repo_id your-username/YourDataset --data_dir /path/to/data --dry_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _matches_any(rel_path: str, patterns: list[str] | None) -> bool:
    """True if rel_path matches any of the glob patterns. Matches on the full
    relative path and on the basename, mirroring huggingface_hub filtering."""
    import fnmatch
    if not patterns:
        return False
    name = rel_path.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(name, pat)
        for pat in patterns
    )


def iter_included_files(
    data_dir: Path,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
):
    """Yield (Path, relative_str) for files that would be uploaded, applying
    the same allow/ignore filtering used by the upload."""
    for f in sorted(data_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(data_dir).as_posix()
        if allow_patterns and not _matches_any(rel, allow_patterns):
            continue
        if _matches_any(rel, ignore_patterns):
            continue
        yield f, rel


def scan_directory(
    data_dir: Path,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    """Return (file_count, total_bytes, top_level_subfolders) for the files that
    would actually be uploaded (after allow/ignore filtering)."""
    file_count = 0
    total_bytes = 0
    for f, _rel in iter_included_files(data_dir, allow_patterns, ignore_patterns):
        file_count += 1
        total_bytes += f.stat().st_size
    subfolders = sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    return file_count, total_bytes, subfolders


def upload_dataset(
    data_dir: Path,
    repo_id: str,
    private: bool = False,
    hf_token: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    commit_message: str | None = None,
) -> None:
    """Create the dataset repo (if needed) and upload the directory contents."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)

    logger.info(f"Creating/verifying dataset repo: {repo_id}")
    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
        token=hf_token,
    )

    logger.info(f"Uploading {data_dir} -> {repo_id} (dataset)")
    # upload_large_folder is resumable and handles many/large files robustly.
    # It preserves the directory structure (subfolders) in the repo.
    api.upload_large_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    logger.info(f"Done. Dataset available at: https://huggingface.co/datasets/{repo_id}")
    logger.info("Load with:")
    logger.info(f'  from datasets import load_dataset')
    logger.info(f'  ds = load_dataset("{repo_id}")')


def main():
    p = argparse.ArgumentParser(
        description="Upload a local data directory (with subfolders) to a HuggingFace dataset repo."
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Local directory to upload. Subfolders are preserved in the repo.")
    p.add_argument("--repo_id", type=str, required=True,
                   help="Existing (or new) dataset repo id, e.g. your-username/YourDataset.")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace access token with write scope. "
                        "If omitted, falls back to the HF_TOKEN env var, then cached login.")
    p.add_argument("--private", action="store_true",
                   help="Create the dataset repo as private if it doesn't exist yet.")
    p.add_argument("--allow_patterns", type=str, nargs="*", default=None,
                   help="Only upload files matching these glob patterns (e.g. '*.parquet' 'train/*').")
    p.add_argument("--ignore_patterns", type=str, nargs="*",
                   default=["__pycache__/*", "*.pyc", ".DS_Store", ".git/*"],
                   help="Skip files matching these glob patterns.")
    p.add_argument("--commit_message", type=str, default=None,
                   help="Commit message for the upload (optional).")
    p.add_argument("--dry_run", action="store_true",
                   help="Scan and report what would be uploaded without pushing.")

    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    file_count, total_bytes, subfolders = scan_directory(
        data_dir, args.allow_patterns, args.ignore_patterns
    )
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Subfolders ({len(subfolders)}): {', '.join(subfolders) if subfolders else '(none)'}")
    logger.info(f"Files to upload: {file_count}  Total size: {_human_size(total_bytes)}")

    if file_count == 0:
        raise SystemExit("No files matched for upload (check --allow_patterns / --ignore_patterns).")

    if args.dry_run:
        logger.info("--dry_run set — not uploading. Files that would be included:")
        shown = 0
        for _f, rel in iter_included_files(data_dir, args.allow_patterns, args.ignore_patterns):
            logger.info(f"  {rel}")
            shown += 1
            if shown >= 20:
                logger.info(f"  ... and {file_count - shown} more")
                break
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    upload_dataset(
        data_dir=data_dir,
        repo_id=args.repo_id,
        private=args.private,
        hf_token=hf_token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
