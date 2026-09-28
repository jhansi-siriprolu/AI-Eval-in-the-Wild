# Day 2 — The code behind log-likelihood scoring (hands-on)

The real, runnable implementation behind Day 1's lesson.

- **`toy_lm.py`** — a tiny, fully-inspectable "language model" (a bigram
  table you can print and edit) plus the actual scoring functions:
  `loglikelihood`, `multiple_choice_score`, `generate_until`, and
  `kv_cache_bytes`. Runs with just `numpy` — no GPU, no `transformers`,
  no downloads.
- **`loglikelihood_vs_generate.ipynb`** — the full walkthrough as a
  runnable Jupyter notebook, with worked examples and exercises. Outputs
  are pre-executed so it reads fine even without running it, but it's
  meant to be run and edited.

## Quickstart

```bash
pip install numpy jupyter

python3 -c "from toy_lm import *; m = ToyLM.hand_built(); \
print(multiple_choice_score(m, '<bos> the capital of france is', \
['paris','berlin','rome','madrid']))"

jupyter notebook loglikelihood_vs_generate.ipynb
```

Read `toy_lm.py` directly once you want to see the real implementation
with comments — it's short enough to read end to end in one sitting.

**Why a toy model instead of a real one?** A real LM's forward pass is:
tokens → embeddings → N transformer layers → logits. The *scoring math*
that eval harnesses (like `lm-evaluation-harness`) do on top of that is
completely independent of what's inside the model — that scoring math is
the actual subject of this lesson. So this swaps the transformer for the
simplest possible thing that still produces "a probability distribution
over the next token given some context": a hand-built bigram table you
can print, edit, and stare at. Everything downstream (loglikelihood
scoring, generate_until, the memory argument) is *exactly* what happens
with a real model — same formulas, same shapes, same reasoning about
GPU memory.
