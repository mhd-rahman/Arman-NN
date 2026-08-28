"""Interactive text generation with ArmanNN.

Usage:
    python chat.py --checkpoint /path/to/step_076000.pt
    python chat.py --checkpoint /path/to/step_076000.pt --device cuda
    python chat.py --checkpoint /path/to/step_076000.pt --max_new_tokens 200 --temperature 0.9
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import generate, load_model


def main():
    parser = argparse.ArgumentParser(description="Interactive generation with ArmanNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name (default: gpt2)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
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
    print(f"Model loaded | params={model.parameter_count():,} | vocab={config.vocab_size} | device={device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size})")
    print()
    print("=" * 60)
    print("ArmanNN Interactive Generation")
    print("Type a prompt and press Enter. Type 'quit' or Ctrl+C to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

        # Decode and print
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n{generated_text}\n")


if __name__ == "__main__":
    main()
