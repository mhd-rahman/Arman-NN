# Arman-NN

A hybrid neural-network architecture combining Attention, Selective SSM, Sparse MoE, Graph Processing, and Neural Memory in one PyTorch model. Includes full production training infrastructure and an OpenAI-compatible API server.

## Architecture

```text
Input Tokens
    │
    ▼
┌─────────────────────────────┐
│  Token Embedding + Position │
└─────────────────────────────┘
    │
    ▼ (× N layers)
┌─────────────────────────────────────────────┐
│  ArmanBlock                                  │
│                                              │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Causal Attention │  │  Selective SSM  │  │
│  │ (Flash Attn 2)   │  │  (parallel scan)│  │
│  └────────┬─────────┘  └────────┬────────┘  │
│           └──── Fusion Gate ─────┘           │
│                     │ + residual             │
│                                              │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │   SwiGLU MLP    │  │   Sparse MoE    │  │
│  │   (dense)       │  │ (top-k experts)  │  │
│  └────────┬─────────┘  └────────┬────────┘  │
│           └──── Path Router ─────┘           │
│                     │ + residual             │
└─────────────────────────────────────────────┘
    │
    ▼ (optional, toggled via config)
┌─────────────────────────────────────────────┐
│  Graph Processor │ Neural Memory │ Router    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  RMSNorm → LM Head (tied)   │
└─────────────────────────────┘
```

## Key Components

| Component | Description |
|-----------|-------------|
| Causal Attention | Multi-head attention with Flash Attention 2 support and KV-cache |
| Selective SSM | Input-dependent linear recurrence with O(√T) parallel scan |
| Fusion Gate | Learned sigmoid gate blending attention and SSM outputs |
| SwiGLU MLP | Standard dense feed-forward (LLaMA-style) |
| Sparse MoE | Top-k expert routing with permute-based batched dispatch |
| Graph Processor | Message-passing GNN for structured relational data |
| Neural Memory | Content-addressed differentiable memory bank |
| Path Routers | Softmax routers selecting between parallel paths |

All components are toggleable via `ArmanConfig` feature switches.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

### Local (single GPU)

```bash
python train.py \
  --dataset_source huggingface \
  --dataset_name stanfordnlp/imdb \
  --text_column text \
  --steps 5000
```

### Distributed (multi-GPU)

```bash
torchrun --nproc_per_node=4 train.py \
  --dataset_source huggingface \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --text_column text \
  --steps 76000
```

### Google Colab / RunPod

Use the notebook at `notebooks/train_colab.ipynb`. It includes:
- Mixed pretraining data (FineWeb-Edu, StarCoder, Wikipedia, OpenWebMath)
- Streaming data pipeline with weighted domain sampling
- bf16 mixed precision + gradient checkpointing
- Automatic checkpoint resume

### AWS SageMaker

```bash
# Submit a managed training job
python sagemaker/launch.py --instance ml.p4d.24xlarge --spot

# Multi-node
python sagemaker/launch.py --instance ml.p4d.24xlarge --instance_count 2 --spot
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/step_022000.pt --dataset toy
```

Metrics computed: loss, perplexity, top-1/5/10 accuracy, MRR.

## Generation

```bash
python generate.py \
  --checkpoint checkpoints/step_022000.pt \
  --prompt "1,2,3,4,5" \
  --max_new_tokens 100 \
  --temperature 0.8
```

Supports top-k, top-p (nucleus), repetition penalty, and KV-cache for fast autoregressive decoding.

## OpenAI-Compatible API Server

Run the model locally with an API that works as a drop-in replacement for OpenAI:

```bash
python serve.py --checkpoint checkpoints/step_022000.pt --port 8000
```

Then use it with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Text completion
response = client.completions.create(
    model="arman-nn",
    prompt="The key to machine learning is",
    max_tokens=100,
)
print(response.choices[0].text)

# Chat completion
response = client.chat.completions.create(
    model="arman-nn",
    messages=[{"role": "user", "content": "Explain neural networks"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

**Endpoints:**
- `POST /v1/completions` — Text completions
- `POST /v1/chat/completions` — Chat completions
- `GET /v1/models` — List available models
- `GET /health` — Health check

## Project Structure

```
Arman-NN/
├── arman/
│   ├── model/
│   │   ├── attention.py      # Multi-head attention + Flash Attention 2
│   │   ├── ssm.py            # Selective SSM with parallel scan
│   │   ├── mlp.py            # SwiGLU feed-forward
│   │   ├── moe.py            # Sparse MoE with permute-based routing
│   │   ├── block.py          # ArmanBlock (attention/SSM fusion + MLP/MoE routing)
│   │   ├── model.py          # ArmanNN top-level model
│   │   ├── graph.py          # Graph message-passing processor
│   │   ├── memory.py         # Neural memory bank
│   │   ├── router.py         # Path router
│   │   └── config.py         # ArmanConfig dataclass
│   └── training/
│       ├── trainer.py         # Full training loop (DDP/FSDP, AMP, checkpointing)
│       ├── scheduler.py       # Warmup + cosine decay LR schedule
│       ├── checkpointing.py   # Save/load/resume with strict=False support
│       ├── distributed.py     # DDP/FSDP setup utilities
│       ├── evaluator.py       # Evaluation pipeline with distributed reduce
│       ├── metrics.py         # Perplexity, top-k accuracy, MRR
│       └── data.py            # HuggingFace + Kaggle dataset loading
├── train.py                   # CLI training script (HF/Kaggle datasets)
├── train_sagemaker.py         # SageMaker training script
├── evaluate.py                # Standalone evaluation script
├── generate.py                # Text generation with KV-cache
├── serve.py                   # OpenAI-compatible API server
├── notebooks/
│   └── train_colab.ipynb      # Colab/RunPod training notebook
├── sagemaker/
│   ├── launch.py              # SageMaker job launcher
│   └── requirements.txt       # Container dependencies
├── tests/
│   └── test_smoke.py          # Smoke test (forward + backward)
└── requirements.txt
```

## Training Infrastructure

| Feature | Status |
|---------|--------|
| Mixed precision (bf16/fp16) | ✅ |
| Flash Attention 2 | ✅ |
| Gradient checkpointing | ✅ |
| KV-cache inference | ✅ |
| Resumable checkpointing | ✅ |
| DDP / FSDP distributed | ✅ |
| Cosine LR schedule | ✅ |
| Gradient accumulation | ✅ |
| Multi-domain streaming data | ✅ |
| Perplexity/accuracy eval | ✅ |
| OpenAI-compatible API | ✅ |
| SageMaker support | ✅ |
| Spot instance resume | ✅ |

## Configuration

All model hyperparameters are controlled via `ArmanConfig`:

```python
from arman.model import ArmanConfig, ArmanNN

config = ArmanConfig(
    vocab_size=50257,
    d_model=1024,
    n_layers=12,
    n_heads=16,
    max_seq_len=1024,
    mlp_hidden=2816,
    expert_hidden=1792,
    n_experts=4,
    moe_top_k=2,
    ssm_state_size=128,
    use_graph=False,     # Toggle graph processor
    use_memory=False,    # Toggle neural memory
)

model = ArmanNN(config)
```

## Tests

```bash
pytest -v
```
