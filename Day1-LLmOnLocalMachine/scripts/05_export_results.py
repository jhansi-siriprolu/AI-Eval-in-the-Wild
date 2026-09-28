#!/usr/bin/env python3
"""
Shapes raw lm-evaluation-harness / mlx_lm.evaluate output into the simple
JSON structure Lesson 08's dashboard (lessons/08-results-dashboard.html)
expects:

    {
      "model": "...",
      "date": "2026-09-24",
      "tasks": {
        "mmlu": {"acc": 0.47, "baseline_random": 0.25},
        ...
      }
    }

Usage:
    python scripts/05_export_results.py
    python scripts/05_export_results.py --input results/raw_eval_output.json
"""
import argparse
import datetime
import glob
import json
import os

# Most lm-eval tasks are 4-way multiple choice (25% random baseline).
# GSM8K is free-form exact-match, so random guessing is ~0%.
RANDOM_BASELINES = {
    "mmlu": 0.25,
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "gsm8k": 0.0,
    "truthfulqa_mc2": 0.5,
    "winogrande": 0.5,
}


def find_input_file(explicit):
    if explicit:
        return explicit
    candidates = sorted(
        glob.glob("results/raw_eval_output.json") + glob.glob("results/*results*.json"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No raw eval output found in results/. Run scripts/04_run_eval_harness.sh first,\n"
            "or pass --input path/to/your/output.json"
        )
    return candidates[0]


def extract_acc(task_result):
    """lm-eval result dicts vary slightly by version; try the common keys."""
    for key in ("acc,none", "acc", "exact_match,none", "exact_match"):
        if key in task_result:
            return task_result[key]
    # Fall back to any key containing 'acc'
    for k, v in task_result.items():
        if "acc" in k and isinstance(v, (int, float)):
            return v
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--output", default="results/results.json")
    args = parser.parse_args()

    input_path = find_input_file(args.input)
    print(f"Reading {input_path} ...")
    with open(input_path) as f:
        raw = json.load(f)

    results_section = raw.get("results", raw)  # lm-eval nests under "results"
    tasks_out = {}
    for task_name, task_result in results_section.items():
        acc = extract_acc(task_result) if isinstance(task_result, dict) else None
        if acc is None:
            continue
        tasks_out[task_name] = {
            "acc": round(acc, 4),
            "baseline_random": RANDOM_BASELINES.get(task_name, 0.25),
        }

    if not tasks_out:
        raise SystemExit(
            "Couldn't find any recognizable task scores in that file. "
            "Open it and check the structure, then adjust extract_acc() above."
        )

    out = {
        "model": raw.get("model_name", args.model),
        "date": datetime.date.today().isoformat(),
        "tasks": tasks_out,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.output}:")
    print(json.dumps(out, indent=2))
    print(
        "\nOpen lessons/08-results-dashboard.html (ideally via "
        "'python -m http.server' from the project root) to see it charted."
    )


if __name__ == "__main__":
    main()
