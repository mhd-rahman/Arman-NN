"""Autoregressive text generation with KV-cache for ArmanNN."""

import argparse
from pathlib import Path

import torch

from arman.model import ArmanConfig, ArmanNN


@torch.no_grad()
def generate(
    model: ArmanNN,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Generate tokens autoregressively using KV-cache.

    Args:
        model: ArmanNN model in eval mode.
        input_ids: (batch, prompt_len) initial token ids.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter).
        top_k: If > 0, keep only top-k logits before sampling.
        top_p: Nucleus sampling threshold (1.0 = disabled).
        repetition_penalty: Penalty for repeated tokens (1.0 = disabled).

    Returns:
        (batch, prompt_len + max_new_tokens) tensor of generated token ids.
    """
    model.eval()
    device = input_ids.device
    generated = input_ids.clone()
    past_key_values = None

    # Prefill: process entire prompt
    out = model(input_ids, past_key_values=None)
    past_key_values = out["present_key_values"]
    logits = out["logits"][:, -1, :]  # (batch, vocab)

    for _ in range(max_new_tokens):
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for b in range(generated.size(0)):
                for token_id in generated[b].unique():
                    logits[b, token_id] /= repetition_penalty

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative prob above threshold
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        generated = torch.cat([generated, next_token], dim=1)

        # Decode step with cache
        out = model(next_token, past_key_values=past_key_values)
        past_key_values = out["present_key_values"]
        logits = out["logits"][:, -1, :]

    return generated


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[ArmanNN, ArmanConfig]:
    """Load model from a training checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in state:
        config_dict = state["config"]
        config = ArmanConfig(**config_dict)
    else:
        raise ValueError("Checkpoint does not contain 'config' key")

    model = ArmanNN(config)

    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        # Legacy format: state dict is the top-level dict
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Generate text with ArmanNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Comma-separated token ids as prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed(args.seed)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device=device)
    print(f"Model loaded | vocab_size={config.vocab_size} | params={model.parameter_count():,}")

    # Prompt
    if args.prompt is not None:
        token_ids = [int(t.strip()) for t in args.prompt.split(",")]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Default: start with token 0 (BOS-like)
        input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

    print(f"Prompt tokens: {input_ids[0].tolist()}")
    print(f"Generating {args.max_new_tokens} tokens...")

    # Generate
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    generated_tokens = output_ids[0].tolist()
    print(f"\nGenerated token ids ({len(generated_tokens)} total):")
    print(generated_tokens)


if __name__ == "__main__":
    main()
