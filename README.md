# AI Evals — Evaluating Language Models on an 8GB M1 Air

An interactive, hands-on project for learning how AI model evaluation works,
using real small open-weight models from Hugging Face, run locally on a
MacBook Air (M1, 8GB RAM, 256GB SSD) via Apple's MLX framework.

## Quick start

1. Open **`lessons/index.html`** in your browser (double-click it, or
   `open lessons/index.html` from Terminal). Start there — it's the course
   home page and links to every lesson in order.
2. Lessons 01–04 and 06–07 are fully interactive in the browser — no setup
   needed. Lesson 05 has you switch to Terminal to actually install MLX and
   run a real model. Lesson 08 charts your real evaluation results once you
   have them.
3. Everything in `scripts/` is the real, runnable code referenced by the
   lessons. See `SETUP.md` for the full environment setup.

## What you'll build

- A working local environment for running Hugging Face models via **MLX**
  (Apple's ML framework, native to Apple Silicon).
- A first working model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
  (~1GB on disk, chosen specifically to fit comfortably in 8GB of RAM —
  see Lesson 03 for the memory math).
- A real evaluation run against standard benchmarks (MMLU, HellaSwag, ARC,
  GSM8K) using `lm-evaluation-harness`, with results charted in
  Lesson 08's dashboard.

## Project layout

```
AI Evals/
├── README.md              this file
├── SETUP.md                step-by-step environment setup
├── requirements.txt         Python dependencies
├── scripts/                 real Python/shell scripts run from Terminal
│   ├── 01_setup_env.sh          create venv + install everything
│   ├── 02_download_and_test.py  download model, run a smoke-test prompt
│   ├── 03_chat_demo.py          interactive chat REPL
│   ├── 04_run_eval_harness.sh   run standard benchmarks
│   └── 05_export_results.py     shape harness output for the dashboard
├── lessons/                 the interactive HTML course — start here
│   ├── index.html
│   ├── 01-intro-model-evaluation.html
│   ├── 02-hugging-face-ecosystem.html
│   ├── 03-quantization-memory-calculator.html
│   ├── 04-tokenization-playground.html
│   ├── 05-running-mlx-model.html
│   ├── 06-perplexity-explained.html
│   ├── 07-benchmarks-quiz.html
│   └── 08-results-dashboard.html
└── results/                  your real evaluation output lands here
    └── results.sample.json     example of the expected shape
```

## Next model ideas

Once Qwen2.5-1.5B is working end-to-end, good next comparisons that still
fit comfortably on 8GB (all in 4-bit MLX builds):

- `mlx-community/SmolLM2-1.7B-Instruct-4bit` — Hugging Face's own small model
- `mlx-community/Llama-3.2-1B-Instruct-4bit` — Meta's smallest Llama 3.2
- `mlx-community/gemma-2-2b-it-4bit` — Google's Gemma 2, 2B instruct

Swap the `--model` argument in `scripts/02_download_and_test.py` or
`scripts/04_run_eval_harness.sh` to try any of these, and compare their
dashboards side by side.
