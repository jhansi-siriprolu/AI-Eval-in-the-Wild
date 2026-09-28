# Day 3 — What does 47% on MMLU actually mean? (interpreting a benchmark score)

How to turn a raw benchmark percentage into an actual judgment: subtract the
random baseline, look past the single averaged number into the 57 subjects
underneath it, and know what the test was and wasn't built to measure.

- **`lesson.html`** — the interactive lesson. Open it directly in a browser
  (double-click, or `open lesson.html`). Includes a live "chance vs. skill"
  slider, the real per-subject breakdown from Day 1's evaluation run
  (sorted low to high, chance-floor line included), a public-model
  comparison table, and a 5-question check-yourself quiz.
- **`mmlu_stats.py`** — the real computation behind Section 5 of the lesson.
  Reads `results.json` (the actual Day 1 eval output, copied in unmodified)
  and prints overall accuracy, points above the 25% baseline, per-subject
  standard deviation, and the 5 hardest/easiest subjects. Runs on the
  standard library alone — no dependencies.
- **`results.json`** — the real MMLU results for
  `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, produced in Day 1. Every
  number quoted in this lesson and its blog post comes from this file.


## Quickstart

```bash
python3 mmlu_stats.py
```

```
Subjects scored:        57
Mean accuracy:          59.6%
Random baseline:        25%
Points above chance:    34.6 pts
Relative lift:          138% better than guessing
Std dev across subjects:16.2%
Min / Max subject:      14% / 90%
```

Then open `lesson.html` in a browser for the interactive version of the
same numbers.

## The core idea in one line

A benchmark score is only meaningful relative to something else — a random
baseline, the spread of subjects it was averaged from, and what the test
was designed to catch. "47% on MMLU" answers none of those questions by
itself; this lesson is about asking them.
