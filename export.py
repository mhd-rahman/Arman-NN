"""Export an ArmanNN checkpoint to HuggingFace format and push to the Hub.

Mirrors the export cell from notebooks/train_colab.ipynb as a standalone CLI.
It loads a training checkpoint, writes HF-compatible files (config, safetensors,
generation config, remote-code modules, the arman/ package, tokenizer, model card)
into an export directory, and optionally pushes everything to the Hub.

Default target repo matches the notebook: mhd-rahman/ArmanNN-Base

Usage:
    # Export the latest checkpoint and push to the default repo:
    python export.py

    # Export a specific checkpoint:
    python export.py --checkpoint checkpoints/step_022000.pt

    # Export to a different repo:
    python export.py --repo_id your-username/YourModel

    # Build the export dir locally without pushing (dry run):
    python export.py --no_push

    # Push without uploading the raw .pt checkpoint (smaller upload):
    python export.py --no_raw_checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch

# Ensure the repo root is importable regardless of invocation location.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arman.training.checkpointing import find_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default repo (matches notebooks/train_colab.ipynb)
DEFAULT_REPO_ID = "mhd-rahman/ArmanNN-Base"


# ---------------------------------------------------------------------------
# Remote-code module sources written into the export dir
# ---------------------------------------------------------------------------

CONFIGURATION_ARMAN_PY = '''"""ArmanNN configuration for HuggingFace transformers."""
from transformers import PretrainedConfig


class ArmanConfig(PretrainedConfig):
    model_type = "arman-nn"

    def __init__(self, vocab_size=50257, d_model=1024, n_layers=12, n_heads=16,
                 max_seq_len=1024, dropout=0.0, ssm_state_size=128, ssm_kernel_size=5,
                 mlp_hidden=2816, n_experts=4, moe_top_k=2, expert_hidden=1792,
                 graph_layers=2, memory_slots=128, memory_top_k=4,
                 use_attention=True, use_ssm=True, use_mlp=True, use_moe=True,
                 use_graph=False, use_memory=False, use_router=True,
                 tie_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.ssm_state_size = ssm_state_size
        self.ssm_kernel_size = ssm_kernel_size
        self.mlp_hidden = mlp_hidden
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k
        self.expert_hidden = expert_hidden
        self.graph_layers = graph_layers
        self.memory_slots = memory_slots
        self.memory_top_k = memory_top_k
        self.use_attention = use_attention
        self.use_ssm = use_ssm
        self.use_mlp = use_mlp
        self.use_moe = use_moe
        self.use_graph = use_graph
        self.use_memory = use_memory
        self.use_router = use_router
        self.tie_embeddings = tie_embeddings
'''

MODELING_ARMAN_PY = '''"""ArmanNN model for HuggingFace transformers."""
import sys
from pathlib import Path

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_arman import ArmanConfig as HFArmanConfig

sys.path.insert(0, str(Path(__file__).parent))
from arman.model.config import ArmanConfig as NativeConfig
from arman.model.model import ArmanNN


class ArmanForCausalLM(PreTrainedModel):
    config_class = HFArmanConfig
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: HFArmanConfig):
        super().__init__(config)
        native_config = NativeConfig(
            vocab_size=config.vocab_size, d_model=config.d_model,
            n_layers=config.n_layers, n_heads=config.n_heads,
            max_seq_len=config.max_seq_len, dropout=config.dropout,
            ssm_state_size=config.ssm_state_size, mlp_hidden=config.mlp_hidden,
            n_experts=config.n_experts, moe_top_k=config.moe_top_k,
            expert_hidden=config.expert_hidden, graph_layers=config.graph_layers,
            memory_slots=config.memory_slots, memory_top_k=config.memory_top_k,
            use_attention=config.use_attention, use_ssm=config.use_ssm,
            use_mlp=config.use_mlp, use_moe=config.use_moe,
            use_graph=config.use_graph, use_memory=config.use_memory,
            use_router=config.use_router, tie_embeddings=config.tie_embeddings,
        )
        self.model = ArmanNN(native_config)

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, **kwargs):
        out = self.model(input_ids, targets=labels, past_key_values=past_key_values)
        return CausalLMOutputWithPast(
            loss=out["loss"], logits=out["logits"],
            past_key_values=out["present_key_values"],
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values}

    def _set_gradient_checkpointing(self, module, value=False):
        if value:
            self.model.enable_gradient_checkpointing()
        else:
            self.model.disable_gradient_checkpointing()
'''


def _model_card(repo_id: str) -> str:
    return f'''---
language: en
license: apache-2.0
tags:
  - arman-nn
  - hybrid-architecture
  - attention
  - ssm
  - moe
  - causal-lm
library_name: transformers
pipeline_tag: text-generation
---

# ArmanNN

A hybrid language model combining Causal Attention, Selective SSM (parallel scan),
Sparse Mixture-of-Experts, with learned fusion gates and path routers.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
'''


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_checkpoint(
    checkpoint_path: Path,
    export_dir: Path,
    repo_id: str,
    tokenizer_name: str = "gpt2",
) -> Path:
    """Write all HuggingFace-format files into export_dir from a checkpoint."""
    from safetensors.torch import save_file as save_safetensors

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state["model"]
    config_dict = state["config"]
    step = state.get("step", 0)
    logger.info(f"Loaded checkpoint at step {step}")

    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json (HF-style)
    hf_config = {
        "architectures": ["ArmanForCausalLM"],
        "model_type": "arman-nn",
        "auto_map": {
            "AutoConfig": "configuration_arman.ArmanConfig",
            "AutoModelForCausalLM": "modeling_arman.ArmanForCausalLM",
        },
        **config_dict,
        "torch_dtype": "bfloat16",
        "training_step": step,
    }
    with open(export_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info("Saved config.json")

    # 2. model weights as safetensors (skip tied lm_head)
    save_state = {}
    for k, v in model_state.items():
        if k == "lm_head.weight":
            continue  # tied to token_embedding
        save_state[f"model.{k}"] = v.contiguous().clone()
    save_safetensors(save_state, str(export_dir / "model.safetensors"))
    logger.info(f"Saved model.safetensors ({sum(p.numel() for p in save_state.values()):,} parameters)")
    del save_state

    # 3. generation_config.json
    gen_config = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
    }
    with open(export_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    logger.info("Saved generation_config.json")

    # 4. remote-code modules
    with open(export_dir / "configuration_arman.py", "w") as f:
        f.write(CONFIGURATION_ARMAN_PY)
    with open(export_dir / "modeling_arman.py", "w") as f:
        f.write(MODELING_ARMAN_PY)
    logger.info("Saved configuration_arman.py and modeling_arman.py")

    # 5. copy the arman/ source package (needed by modeling_arman.py at runtime)
    src_pkg = _REPO_ROOT / "arman"
    dst_pkg = export_dir / "arman"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(
        src_pkg, dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    logger.info("Copied arman/ package")

    # 6. tokenizer files
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.save_pretrained(str(export_dir))
    logger.info(f"Saved tokenizer ({tokenizer_name})")

    # 7. model card
    with open(export_dir / "README.md", "w") as f:
        f.write(_model_card(repo_id))
    logger.info("Saved README.md (model card)")

    logger.info(f"Export complete. Files in: {export_dir}/")
    return export_dir


def push_to_hub(
    export_dir: Path,
    repo_id: str,
    checkpoint_path: Path | None = None,
    private: bool = False,
    hf_token: str | None = None,
) -> None:
    """Push the export directory (and optionally the raw checkpoint) to the Hub."""
    from huggingface_hub import HfApi, create_repo

    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
    api = HfApi(token=hf_token)
    create_repo(repo_id, exist_ok=True, repo_type="model", private=private, token=hf_token)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info("Uploaded export folder")

    if checkpoint_path is not None:
        logger.info(f"Uploading raw checkpoint: {checkpoint_path}")
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"checkpoints/{checkpoint_path.name}",
            repo_id=repo_id,
            repo_type="model",
        )

    logger.info(f"Pushed! Model available at: https://huggingface.co/{repo_id}")
    logger.info("Load with:")
    logger.info(f'  model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)')
    logger.info(f'  tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export an ArmanNN checkpoint to HuggingFace and push to the Hub.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint .pt path. Default: latest in --checkpoint_dir.")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory to search for the latest checkpoint if --checkpoint is not given.")
    p.add_argument("--export_dir", type=str, default="hf_export",
                   help="Local directory to write HF-format files into (default: hf_export).")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                   help=f"Target Hub repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="Tokenizer to bundle with the model (default: gpt2).")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace access token with write scope. "
                        "Falls back to the HF_TOKEN env var, then to cached login.")
    p.add_argument("--private", action="store_true",
                   help="Create the Hub repo as private if it doesn't exist yet.")
    p.add_argument("--no_push", action="store_true",
                   help="Build the export directory locally but do not push to the Hub.")
    p.add_argument("--no_raw_checkpoint", action="store_true",
                   help="Do not upload the raw .pt checkpoint alongside the HF files.")

    args = p.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    else:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest is None:
            raise SystemExit(
                f"No checkpoint found in {args.checkpoint_dir}. "
                f"Pass one explicitly with --checkpoint."
            )
        checkpoint_path = latest
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    export_dir = Path(args.export_dir)
    export_checkpoint(
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
        repo_id=args.repo_id,
        tokenizer_name=args.tokenizer,
    )

    if args.no_push:
        logger.info("--no_push set — skipping Hub upload. Export directory is ready locally.")
        return

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    push_to_hub(
        export_dir=export_dir,
        repo_id=args.repo_id,
        checkpoint_path=None if args.no_raw_checkpoint else checkpoint_path,
        private=args.private,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
