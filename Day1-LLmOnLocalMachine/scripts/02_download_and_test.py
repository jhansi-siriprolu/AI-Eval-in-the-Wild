#!/usr/bin/env python3
"""
Downloads (on first run) and smoke-tests the course's default model:
mlx-community/Qwen2.5-1.5B-Instruct-4bit

Usage:
    python scripts/02_download_and_test.py
    python scripts/02_download_and_test.py --model mlx-community/SmolLM2-1.7B-Instruct-4bit
    python scripts/02_download_and_test.py --prompt "Explain perplexity in one sentence."
"""
import argparse
import time

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_PROMPT = (
    "In one sentence, explain why quantization helps run AI models on laptops."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        raise SystemExit(
            "mlx-lm isn't installed yet. Run:\n"
            "    bash scripts/01_setup_env.sh\n"
            "or:\n"
            "    pip install -r requirements.txt"
        )

    print(f"Loading {args.model} (downloads on first run, then cached)...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    # Use the model's chat template if it has one (instruct models do).
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    else:
        prompt = args.prompt

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=True,  # prints tokens/sec + peak memory, exactly what Lesson 05 shows
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
