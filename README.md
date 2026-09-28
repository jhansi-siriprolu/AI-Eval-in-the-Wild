# AI Eval in the Wild

A daily, hands-on series on how AI model evaluation actually works — run
locally, on ordinary hardware (an 8GB M1 MacBook Air), with real open-weight
models and real benchmark numbers. No cloud GPU, no API keys, nothing taken
on faith: every lesson ships the code and the data behind its claims.

Each day lives in its own folder: an interactive `lesson.html` you open
straight in a browser, the real runnable code, and a Medium-ready `blog.md`
write-up of the same material.

## The series so far

| Day | Topic | Folder |
|---|---|---|
| 1 | Running a real LLM locally on an 8GB M1 Air — Hugging Face, MLX, quantization, and a first live evaluation | [`Day1-LLmOnLocalMachine/`](./Day1-LLmOnLocalMachine) |
| 2 | Log-likelihood scoring vs. `generate_until` — how MMLU/HellaSwag/ARC score answers without generating a word, and why that's what lets them run on 8GB | [`Day2-loglikelihood-code/`](./Day2-loglikelihood-code) |
| 3 | What does 47% on MMLU actually mean? — random baselines, per-subject variance, and how to read a benchmark score critically | [`Day3-mmlu-score-interpretation/`](./Day3-mmlu-score-interpretation) |

More days land as the series continues — check this table for the latest.

## How each day is structured

```
DayN-topic-name/
├── README.md       what this lesson covers, quickstart
├── lesson.html      the interactive lesson — open directly in a browser
├── <code/data>       the real, runnable code and/or data behind the lesson
└── blog.md           Medium-ready write-up of the same lesson
```

Nothing in `lesson.html` requires a server or a build step — every page is
a single self-contained file. Where a lesson needs real numbers (evaluation
results, model output), the raw data ships alongside it rather than being
hardcoded, so you can regenerate or verify every claim yourself.

## Prerequisites

- Python 3.10+
- For lessons that run a real model: a Mac with Apple Silicon and 8GB+ RAM
  (see `SETUP.md` for the full MLX environment setup)
- For lessons that only analyze existing results (like Day 3): the standard
  library is enough — no GPU or model download needed

## Try it out

```bash
git clone https://github.com/jhansi-siriprolu/AI-Eval-in-the-Wild.git
cd AI-Eval-in-the-Wild
```

Then open any `DayN-.../lesson.html` in a browser, or follow that day's own
`README.md` to run its code.

**Repo:** [github.com/jhansi-siriprolu/AI-Eval-in-the-Wild](https://github.com/jhansi-siriprolu/AI-Eval-in-the-Wild)

## License

See [`LICENSE`](./LICENSE).
