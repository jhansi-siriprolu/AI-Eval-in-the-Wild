#!/usr/bin/env python3
import sys, os, json, argparse

class SafeEncoder(json.JSONEncoder):
    """Strip anything json can't handle — functions, numpy types, etc."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if callable(obj): return f"<function {getattr(obj, '__name__', str(obj))}>"
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--tasks",  default="mmlu,hellaswag,arc_easy")
    parser.add_argument("--limit",  type=int, default=50)
    parser.add_argument("--out",    default="results/raw_eval_output.json")
    args = parser.parse_args()

    import mlx_lm.evaluate   # registers mlxlm into lm_eval
    import lm_eval

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = lm_eval.simple_evaluate(
        model="mlxlm",
        model_args="path_or_hf_repo=" + args.model + ",use_chat_template=True,batch_size=1",
        tasks=tasks,
        limit=args.limit,
        log_samples=False,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, cls=SafeEncoder)

    print(f"\nWrote {args.out}")
    print("Scores:")
    for task, vals in results.get("results", {}).items():
        acc = vals.get("acc,none", vals.get("acc", vals.get("exact_match,none", "?")))
        print(f"  {task}: {acc}")

if __name__ == "__main__":
    main()
