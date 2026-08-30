"""Stage-2 continued-pretraining launcher for ArmanNN.

This is a thin wrapper around ``training/train.py``. It starts a NEW training
stage from a stage-1 checkpoint: it loads only the model weights and begins at
step 0 with a fresh optimizer and a fresh warmup+cosine LR schedule — it does
NOT inherit the optimizer/scheduler/step from the source checkpoint.

Stage-2 defaults applied here (all overridable on the command line):
    --lr            5e-5      (lower LR for continued pretraining)
    --warmup_steps  2000
    --min_lr_ratio  0.1       (cosine decays down to 10% of peak)
    --checkpoint_dir checkpoints_2
    --dataset_name   ../../data/pretrained_2   (the stage-2 mix)
    --metrics_log    training_metrics_2.csv

The scheduler is warmup + cosine decay (arman.training.scheduler), which is what
train.py always uses — so "cosine" is satisfied by the existing scheduler.

Usage:
    # Continue pretraining from the latest stage-1 checkpoint:
    python training/train2.py --init_from checkpoints/step_022000.pt \
        --dataset_name ../../data/pretrained_2 --steps 30000

    # Distributed:
    torchrun --nproc_per_node=4 training/train2.py \
        --init_from checkpoints/step_022000.pt --dataset_name ../../data/pretrained_2

Any flag understood by train.py can be passed and will override the stage-2
defaults below (e.g. --lr, --warmup_steps, --batch_size, --steps, --eval_data).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.train import main as train_main


# Stage-2 defaults: (flag, value). Only injected if the user did not pass the
# flag explicitly. Everything here can be overridden on the command line.
STAGE2_DEFAULTS = {
    "--lr": "5e-5",
    "--warmup_steps": "2000",
    "--min_lr_ratio": "0.1",
    "--checkpoint_dir": "checkpoints_2",
    "--metrics_log": "training_metrics_2.csv",
}


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def main():
    argv = sys.argv[1:]

    # Inject stage-2 defaults for any flag the user didn't specify.
    injected = []
    for flag, value in STAGE2_DEFAULTS.items():
        if not _has_flag(argv, flag):
            injected += [flag, value]

    # --init_from is required for a stage-2 run (it's what makes this a fresh
    # stage rather than a plain resume). Fail early with a clear message.
    if not _has_flag(argv, "--init_from"):
        raise SystemExit(
            "train2.py requires --init_from <stage-1 checkpoint.pt> so it can load the "
            "model weights and start a fresh stage. Example:\n"
            "  python training/train2.py --init_from checkpoints/step_022000.pt "
            "--dataset_name ../../data/pretrained_2 --steps 30000"
        )

    # dataset_name is required by train.py; give a stage-2 default if omitted.
    if not _has_flag(argv, "--dataset_name"):
        injected += ["--dataset_name", "../../data/pretrained_2"]

    # Rebuild argv so train.py's argparse sees the merged set. User-provided
    # args come last so argparse's "last wins" keeps explicit overrides.
    sys.argv = [sys.argv[0]] + injected + argv

    train_main()


if __name__ == "__main__":
    main()
