"""Dataset loading from HuggingFace Hub and Kaggle with tokenization."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """A tokenized text dataset for next-token prediction.

    Takes pre-tokenized sequences and returns (input_ids, targets) pairs
    where targets are input_ids shifted by one position.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        """
        Args:
            token_ids: 1D tensor of all token ids concatenated.
            seq_len: Length of each training sequence.
        """
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Number of complete sequences we can form
        self.n_samples = (len(token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.token_ids[start : start + self.seq_len]
        y = self.token_ids[start + 1 : start + self.seq_len + 1]
        return x, y


def _get_tokenizer(tokenizer_name: str):
    """Load a HuggingFace tokenizer by name."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_texts(texts: list[str], tokenizer, max_length: int = 0) -> torch.Tensor:
    """Tokenize a list of texts and concatenate into a single 1D tensor."""
    all_ids = []
    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    logger.info(f"Tokenized {len(texts)} texts into {len(all_ids):,} tokens")
    return torch.tensor(all_ids, dtype=torch.long)


def load_huggingface_dataset(
    dataset_name: str,
    tokenizer_name: str = "gpt2",
    seq_len: int = 1024,
    split: str = "train",
    text_column: str = "text",
    max_samples: int = 0,
    subset: str | None = None,
) -> TextDataset:
    """Load a dataset from HuggingFace Hub and tokenize it.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. 'wikitext/wikitext-2-raw-v1',
                      'openwebtext', 'allenai/c4').
        tokenizer_name: HuggingFace tokenizer to use (default: 'gpt2').
        seq_len: Training sequence length.
        split: Dataset split to load ('train', 'validation', 'test').
        text_column: Name of the text column in the dataset.
        max_samples: Max number of raw samples to load (0 = all).
        subset: Dataset subset/config name if applicable.

    Returns:
        TextDataset ready for training.
    """
    from datasets import load_dataset

    logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split}, subset={subset})")

    load_kwargs = {"split": split}
    if subset:
        ds = load_dataset(dataset_name, subset, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"Loaded {len(ds)} samples from HuggingFace")

    # Extract text
    if text_column not in ds.column_names:
        available = ds.column_names
        raise ValueError(
            f"Column '{text_column}' not found in dataset. Available: {available}"
        )

    texts = ds[text_column]

    # Tokenize
    tokenizer = _get_tokenizer(tokenizer_name)
    token_ids = _tokenize_texts(texts, tokenizer)

    logger.info(f"Created dataset with {(len(token_ids) - 1) // seq_len} sequences of length {seq_len}")
    return TextDataset(token_ids, seq_len)


def load_kaggle_dataset(
    dataset_name: str,
    tokenizer_name: str = "gpt2",
    seq_len: int = 1024,
    text_column: str = "text",
    file_pattern: str = "*.csv",
    max_samples: int = 0,
    download_dir: str = "./data/kaggle",
) -> TextDataset:
    """Load a dataset from Kaggle and tokenize it.

    Requires the Kaggle API to be configured (KAGGLE_USERNAME and KAGGLE_KEY
    environment variables, or ~/.kaggle/kaggle.json).

    Args:
        dataset_name: Kaggle dataset identifier (e.g. 'username/dataset-name').
        tokenizer_name: HuggingFace tokenizer to use (default: 'gpt2').
        seq_len: Training sequence length.
        text_column: Name of the text column in the CSV/parquet files.
        file_pattern: Glob pattern for data files after download.
        max_samples: Max number of raw samples to load (0 = all).
        download_dir: Local directory to download Kaggle data into.

    Returns:
        TextDataset ready for training.
    """
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    download_path = Path(download_dir) / dataset_name.replace("/", "_")
    download_path.mkdir(parents=True, exist_ok=True)

    # Download if not already present
    data_files = list(download_path.glob(file_pattern))
    if not data_files:
        logger.info(f"Downloading Kaggle dataset: {dataset_name} -> {download_path}")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=str(download_path), unzip=True)
        data_files = list(download_path.glob(file_pattern))

    if not data_files:
        # Try parquet files as fallback
        data_files = list(download_path.glob("*.parquet"))

    if not data_files:
        raise FileNotFoundError(
            f"No data files found in {download_path} matching '{file_pattern}' or '*.parquet'"
        )

    logger.info(f"Found {len(data_files)} data files in {download_path}")

    # Load all files into a single DataFrame
    dfs = []
    for f in sorted(data_files):
        if f.suffix == ".csv":
            dfs.append(pd.read_csv(f))
        elif f.suffix == ".parquet":
            dfs.append(pd.read_parquet(f))
        elif f.suffix in (".json", ".jsonl"):
            dfs.append(pd.read_json(f, lines=f.suffix == ".jsonl"))
        else:
            logger.warning(f"Skipping unsupported file: {f}")

    if not dfs:
        raise ValueError("No supported data files could be loaded")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from Kaggle dataset")

    if text_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {available}"
        )

    if max_samples > 0:
        df = df.head(max_samples)

    texts = df[text_column].dropna().tolist()

    # Tokenize
    tokenizer = _get_tokenizer(tokenizer_name)
    token_ids = _tokenize_texts(texts, tokenizer)

    logger.info(f"Created dataset with {(len(token_ids) - 1) // seq_len} sequences of length {seq_len}")
    return TextDataset(token_ids, seq_len)


def load_dataset_from_source(
    source: str,
    dataset_name: str,
    tokenizer_name: str = "gpt2",
    seq_len: int = 1024,
    split: str = "train",
    text_column: str = "text",
    max_samples: int = 0,
    subset: str | None = None,
    **kwargs,
) -> TextDataset:
    """Unified entry point for loading datasets from any supported source.

    Args:
        source: One of 'huggingface' or 'kaggle'.
        dataset_name: Dataset identifier for the chosen source.
        tokenizer_name: HuggingFace tokenizer name.
        seq_len: Training sequence length.
        split: Dataset split (HuggingFace only).
        text_column: Column containing text data.
        max_samples: Limit on raw samples (0 = all).
        subset: Dataset subset/config (HuggingFace only).
        **kwargs: Additional source-specific arguments.

    Returns:
        TextDataset ready for training.
    """
    source = source.lower().strip()

    if source == "huggingface":
        return load_huggingface_dataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            split=split,
            text_column=text_column,
            max_samples=max_samples,
            subset=subset,
        )
    elif source == "kaggle":
        return load_kaggle_dataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            text_column=text_column,
            max_samples=max_samples,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset source: '{source}'. Use 'huggingface' or 'kaggle'.")
