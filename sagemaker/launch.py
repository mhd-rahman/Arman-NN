"""
SageMaker Training Job Launcher for ArmanNN

Submits a managed training job to AWS SageMaker.

Prerequisites:
    - AWS credentials configured (aws configure)
    - SageMaker execution role with S3 access
    - pip install sagemaker boto3

Usage:
    python sagemaker/launch.py

    # With spot instances (70% cheaper, auto-resumes from checkpoint):
    python sagemaker/launch.py --spot

    # Custom instance:
    python sagemaker/launch.py --instance ml.p5.48xlarge --spot

    # Multi-node:
    python sagemaker/launch.py --instance ml.p4d.24xlarge --instance_count 2
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    parser = argparse.ArgumentParser(description="Launch ArmanNN SageMaker Training Job")

    # Infrastructure
    parser.add_argument("--instance", type=str, default="ml.p4d.24xlarge",
                        help="Instance type (default: ml.p4d.24xlarge = 8×A100 40GB)")
    parser.add_argument("--instance_count", type=int, default=1,
                        help="Number of instances (for multi-node)")
    parser.add_argument("--spot", action="store_true",
                        help="Use spot instances (~70%% cheaper, auto-resumes)")
    parser.add_argument("--max_run", type=int, default=72 * 3600,
                        help="Max run time in seconds (default: 72h)")
    parser.add_argument("--max_wait", type=int, default=96 * 3600,
                        help="Max wait time for spot (default: 96h)")

    # SageMaker config
    parser.add_argument("--role", type=str, default=None,
                        help="SageMaker execution role ARN (auto-detected if not set)")
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="S3 bucket for output (uses default SageMaker bucket if not set)")
    parser.add_argument("--job_name", type=str, default="arman-nn-training",
                        help="Training job name prefix")

    # Hyperparameters (override defaults in train_sagemaker.py)
    parser.add_argument("--total_steps", type=int, default=76000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated datasets")

    args = parser.parse_args()

    # Session
    session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    bucket = args.s3_bucket or session.default_bucket()

    # Checkpoint S3 location
    checkpoint_s3_uri = f"s3://{bucket}/arman-nn/checkpoints"
    output_path = f"s3://{bucket}/arman-nn/output"

    # Hyperparameters for the training script
    hyperparameters = {
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "save_every": 2000,
        "eval_every": 2000,
        "log_every": 50,
        "gradient_checkpointing": "",  # flag-style arg
    }
    if args.hf_token:
        hyperparameters["hf_token"] = args.hf_token

    # Distribution config for multi-GPU
    distribution = {
        "torch_distributed": {
            "enabled": True,
        }
    }

    # Estimator
    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=".",  # Upload entire project
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance,
        framework_version="2.4.0",
        py_version="py311",
        hyperparameters=hyperparameters,
        distribution=distribution,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path="/opt/ml/checkpoints",
        max_run=args.max_run,
        # Spot instance configuration
        use_spot_instances=args.spot,
        max_wait=args.max_wait if args.spot else None,
        # Environment
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        },
        # Tags for cost tracking
        tags=[
            {"Key": "Project", "Value": "ArmanNN"},
            {"Key": "Team", "Value": "Research"},
        ],
        # Dependencies
        requirements_file="sagemaker/requirements.txt",
    )

    print(f"Launching SageMaker Training Job:")
    print(f"  Instance: {args.instance} × {args.instance_count}")
    print(f"  Spot: {args.spot}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} grad_accum × {8 * args.instance_count} GPUs")
    print(f"  Checkpoints: {checkpoint_s3_uri}")
    print(f"  Output: {output_path}")
    print()

    estimator.fit(job_name=args.job_name, wait=False)
    print(f"Job submitted! Monitor at: https://console.aws.amazon.com/sagemaker/")
    print(f"Job name: {estimator.latest_training_job.name}")


if __name__ == "__main__":
    main()
