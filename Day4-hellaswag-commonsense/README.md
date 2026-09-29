# Day 4 — HellaSwag and the Art of Testing Common Sense

*Why finishing a sentence correctly requires more than pattern matching — and how we measure it.*

One lesson, three formats that follow the **same eight sections in the same order**:

| File | What it is |
|---|---|
| `lesson.html` | The blog-style lesson. Open it directly in a browser (no server needed). Interactive: try the quiz, the length-trap calculator and the softmax playground. |
| `hellaswag_lesson.ipynb` | The notebook: every section as explanation + runnable code, with outputs saved. **Generated from `hellaswag_demo.py`**, so the two can never drift apart. |
| `hellaswag_demo.py` | The runnable script (also opens as cells in VS Code / Cursor thanks to the `# %%` markers). |
| `adversarial_filtering_curve.png`, `score_context.png` | Charts the script draws. |

## The eight sections

1. **Anatomy of one question** — activity label, context, four endings, label.
2. **The maths by hand** — probability → logarithm → sum → softmax, and the length trap.
3. **A real model takes the test** — GPT-2 scores endings three ways; scores 1,000 random validation questions.
4. **Cheap tricks fail** — longest ending, word overlap, ending-only n-gram classifier.
5. **Adversarial Filtering in miniature** — a toy generator vs discriminator game.
6. **What does 55% mean?** — gap to humans, margin of error, why common sense is harder than facts.
7. **Reading the mistakes** — real wrong answers.
8. **What HellaSwag still misses** — limits, saturation, contamination.

## Quickstart

```bash
pip install torch transformers datasets scikit-learn matplotlib numpy
python hellaswag_demo.py                      # small model (124 million parameters), 1,000 questions
python hellaswag_demo.py --limit 200          # quick run (a few minutes); fewer questions = a wobblier score
python hellaswag_demo.py --model gpt2-xl      # 1.5 billion parameters; use a machine with 8 GB or more free
python hellaswag_demo.py --filter_rounds 8    # longer Adversarial Filtering game
```

Works offline for Sections 1–3 using an embedded copy of the worked example (the model weights still need to
have been downloaded once). Sections 4–8 need the dataset from Hugging Face (`Rowan/hellaswag`).

On Apple Silicon the script uses the graphics chip (`mps`) automatically. Numbers quoted in `lesson.html` come from
a 1,000-question run of `gpt2` on a CPU with seed 0; half-precision on `mps` can shift a score by a fraction of a point.

## Regenerating the notebook

The notebook is produced from the script with [jupytext](https://jupytext.readthedocs.io):

```bash
pip install jupytext nbconvert ipykernel
jupytext --to ipynb hellaswag_demo.py -o hellaswag_lesson.ipynb
jupyter nbconvert --to notebook --execute --inplace hellaswag_lesson.ipynb
```
