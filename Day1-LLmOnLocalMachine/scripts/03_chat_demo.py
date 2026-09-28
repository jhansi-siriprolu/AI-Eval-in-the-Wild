#!/usr/bin/env python3
"""
A minimal interactive chat REPL. Loads the model once, then keeps it
resident in memory across turns so conversation feels responsive.

Usage:
    python scripts/03_chat_demo.py
    python scripts/03_chat_demo.py --model mlx-community/SmolLM2-1.7B-Instruct-4bit

Type 'exit' or Ctrl-D to quit.
"""
import argparse

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--system",
        default="You are a concise, helpful assistant running locally on an M1 MacBook Air.",
    )
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        raise SystemExit(
            "mlx-lm isn't installed yet. Run: bash scripts/01_setup_env.sh"
        )

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    print("Loaded. Type your message (or 'exit' to quit).\n")

    history = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)

        print("Model: ", end="", flush=True)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        print(response.strip(), "\n")
        history.append({"role": "assistant", "content": response.strip()})


if __name__ == "__main__":
    main()
