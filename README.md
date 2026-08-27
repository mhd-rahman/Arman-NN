# Arman-NN v0.1

A research-first hybrid neural-network prototype that combines all of the mechanisms discussed for the initial Arman brain architecture in one PyTorch model.

## Included

- Token + positional embeddings
- Causal multi-head self-attention
- Selective recurrent SSM-style branch
- Learned Attention/SSM fusion gate
- SwiGLU dense MLP
- Sparse top-k Mixture of Experts with routing auxiliary loss
- Graph message-passing processor + graph-to-sequence cross attention
- Differentiable content-addressed neural memory
- Local and global neural routers
- RMSNorm + residual connections
- Weight-tied language-model output head
- Feature switches through `ArmanConfig`

> The SSM and memory components are original small research implementations inspired by the corresponding architectural ideas; they are not reproductions of Mamba or Titans.

## Architecture

```text
Tokens
  |
Embedding + Position
  |
  +---------------- Arman Block x N ----------------+
  | RMSNorm                                         |
  |     +-------------+                             |
  |     |             |                             |
  | Attention       Selective SSM                   |
  |     |             |                             |
  |     +-- learned fusion gate                     |
  |              |                                  |
  |           residual                              |
  |              |                                  |
  |           RMSNorm                               |
  |              |                                  |
  |         local router                            |
  |          /       \                              |
  |       SwiGLU   Sparse MoE                       |
  |          \       /                              |
  |           residual                              |
  +-------------------------------------------------+
                 |
       +---------+----------+
       |                    |
   Graph branch         Neural memory
       |                    |
       +------ global router+
                 |
             RMSNorm
                 |
             LM Head
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

```bash
pytest -q
```

## Train the toy task

```bash
python train.py --steps 100
```

This intentionally trains on a synthetic next-token sequence task. Its purpose is to prove forward/backward training of the complete architecture before introducing a tokenizer and real corpora.

## Next research milestones

1. Establish a parameter-matched Transformer baseline.
2. Add real tokenizer and streaming text datasets.
3. Replace the reference recurrent SSM with a parallel scan / optimized selective SSM kernel.
4. Add causal/persistent memory update semantics across sequences.
5. Build graph extraction/training objectives rather than requiring graphs as supplied inputs.
6. Add router diagnostics, expert load balancing and capacity controls.
7. Run ablations for Attention, SSM, MoE, Graph and Memory.
8. Scale 5M -> 20M -> 100M parameters only after the hybrid demonstrates measurable benefit.
