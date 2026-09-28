"""
Day 3 — turning a raw MMLU results.json into the numbers this lesson talks about:
overall accuracy vs. the random baseline, per-subject spread, and the "z-score
above chance" framing used to judge whether a score is actually meaningful.

Runs on the *real* results.json produced in Day 1 (Qwen2.5-1.5B-Instruct-4bit),
copied into this folder. No downloads, no GPU — just stdlib + the json file.
"""

import json
import statistics
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.json"
RANDOM_BASELINE = 0.25  # MMLU is always 4-way multiple choice


def load_subject_scores(path: Path = RESULTS_PATH) -> dict[str, float]:
    """Return {subject_name: accuracy} for every mmlu_<subject> key,
    skipping the rolled-up category keys (mmlu, mmlu_stem, mmlu_other, ...)."""
    data = json.loads(path.read_text())
    rollups = {"mmlu", "mmlu_stem", "mmlu_other", "mmlu_social_sciences", "mmlu_humanities"}
    return {
        name.replace("mmlu_", ""): task["acc"]
        for name, task in data["tasks"].items()
        if name.startswith("mmlu_") and name not in rollups
    }


def summarize(scores: dict[str, float]) -> None:
    values = list(scores.values())
    overall = statistics.mean(values)
    hardest = sorted(scores.items(), key=lambda kv: kv[1])[:5]
    easiest = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

    print(f"Subjects scored:        {len(values)}")
    print(f"Mean accuracy:          {overall:.1%}")
    print(f"Random baseline:        {RANDOM_BASELINE:.0%}")
    print(f"Points above chance:    {(overall - RANDOM_BASELINE) * 100:.1f} pts")
    print(f"Relative lift:          {(overall / RANDOM_BASELINE - 1) * 100:.0f}% better than guessing")
    print(f"Std dev across subjects:{statistics.pstdev(values):.1%}")
    print(f"Min / Max subject:      {min(values):.0%} / {max(values):.0%}")
    print()
    print("5 hardest subjects for this model:")
    for name, acc in hardest:
        flag = "  <- near or below chance" if acc <= RANDOM_BASELINE + 0.05 else ""
        print(f"  {name:38s} {acc:5.0%}{flag}")
    print()
    print("5 easiest subjects for this model:")
    for name, acc in easiest:
        print(f"  {name:38s} {acc:5.0%}")


if __name__ == "__main__":
    scores = load_subject_scores()
    summarize(scores)
