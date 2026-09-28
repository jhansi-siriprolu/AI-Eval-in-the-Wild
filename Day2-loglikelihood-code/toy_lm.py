"""
toy_lm.py — a tiny, fully-inspectable "language model" used to teach
log-likelihood scoring (used by MMLU / HellaSwag / ARC) vs. generate_until
(used by GSM8K), without needing a GPU, torch, or a real 7B-parameter model.

Why a toy model instead of a real one?
---------------------------------------
A real LM's forward pass is: tokens -> embeddings -> 32 transformer layers ->
logits. The *scoring math* that eval harnesses (like lm-evaluation-harness)
do on top of that is completely independent of what's inside the model.
That scoring math is the actual subject of this lesson. So we replace the
32-layer transformer with the simplest possible thing that still produces
"a probability distribution over the next token given some context": a
hand-built bigram table you can print, edit, and stare at.

Everything downstream (loglikelihood scoring, generate_until, the memory
argument) is *exactly* what happens with a real model — same formulas, same
shapes, same reasoning about GPU memory. Only the box that produces
P(next_token | context) has been shrunk from "175B parameters" down to
"a Python dict you can read in one screen."
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. A tiny word-level vocabulary and a hand-built bigram model
# ---------------------------------------------------------------------------

VOCAB = [
    "<bos>", "the", "capital", "of", "france", "germany", "italy", "is",
    "paris", "berlin", "rome", "madrid", "a", "city", "in", "europe",
    "answer", ":", "yes", "no", "42", "5", "7", "12",
]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for i, t in enumerate(VOCAB)}
V = len(VOCAB)


def tokenize(text: str) -> List[int]:
    """Dead-simple whitespace tokenizer into our fixed toy vocabulary."""
    words = text.lower().replace(":", " : ").split()
    ids = []
    for w in words:
        if w not in TOK2ID:
            raise ValueError(f"toy vocab doesn't know the word {w!r}. "
                              f"Known words: {VOCAB}")
        ids.append(TOK2ID[w])
    return ids


@dataclass
class ToyLM:
    """
    A bigram language model: P(next_token | previous_token) only
    (a real transformer conditions on the *entire* preceding context via
    attention; we condition on just the last token so the whole model is a
    V x V matrix you can print). The scoring code below (loglikelihood,
    generate_until) does not care about this difference — it only calls
    `next_token_logits(context_ids)` and treats the model as a black box,
    exactly like a real eval harness treats a real LM as a black box.
    """
    logits: np.ndarray = field(default_factory=lambda: np.zeros((V, V)))

    @classmethod
    def hand_built(cls) -> "ToyLM":
        """
        Build logits by hand so that:
          "the capital of france is" -> strongly prefers "paris"
          "the capital of germany is" -> strongly prefers "berlin"
          "the capital of italy is" -> strongly prefers "rome"
        and other continuations get low (but nonzero) probability.
        We do this by setting raw logits (pre-softmax scores); big positive
        number = "model is confident", 0 = neutral, negative = "unlikely".
        """
        logits = np.full((V, V), -2.0)  # mild background "unlikely" score

        def boost(prev_word, next_word, score):
            logits[TOK2ID[prev_word], TOK2ID[next_word]] = score

        # bigram continuations that make short factual sentences plausible
        boost("<bos>", "the", 3.0)
        boost("the", "capital", 2.0)
        boost("capital", "of", 4.0)
        boost("of", "france", 1.5)
        boost("of", "germany", 1.5)
        boost("of", "italy", 1.5)
        boost("france", "is", 4.0)
        boost("germany", "is", 4.0)
        boost("italy", "is", 4.0)

        # THE key modeling choice: how confident is "is" -> capital city?
        boost("is", "paris", 5.0)
        boost("is", "berlin", 3.0)
        boost("is", "rome", 2.5)
        boost("is", "madrid", 0.5)   # plausible-sounding wrong answer
        boost("is", "a", 1.0)

        boost("a", "city", 3.0)
        boost("city", "in", 3.0)
        boost("in", "europe", 3.0)

        boost("answer", ":", 5.0)
        boost(":", "42", 1.0)
        boost(":", "5", 1.0)
        boost(":", "7", 2.5)   # model "believes" 7 is the right arithmetic answer
        boost(":", "12", 1.0)
        boost(":", "yes", 1.0)
        boost(":", "no", 1.0)

        return cls(logits=logits)

    def next_token_logits(self, context_ids: List[int]) -> np.ndarray:
        """Return raw logits over the vocab for the token *after* context.
        A real model: feed all context_ids through the transformer, read off
        the logits at the last position. Here: look up the last token's row.
        """
        prev = context_ids[-1]
        return self.logits[prev]

    def next_token_logprobs(self, context_ids: List[int]) -> np.ndarray:
        """log P(next=v | context) for every v in the vocab, via log-softmax."""
        z = self.next_token_logits(context_ids)
        return log_softmax(z)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(softmax(z)). This one line is the entire
    bridge between 'raw model outputs' and 'probabilities that sum to 1'."""
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


# ---------------------------------------------------------------------------
# 2. loglikelihood scoring — how MMLU / HellaSwag / ARC actually work
# ---------------------------------------------------------------------------

def token_logprobs_for_continuation(
    model: ToyLM, context_ids: List[int], continuation_ids: List[int]
) -> List[float]:
    """
    Core primitive. Given a context and a *specific* candidate continuation,
    compute log P(token | everything before it) for each token in the
    continuation, one at a time, walking the sequence left to right.

    This is a single forward pass over [context + continuation] in a real
    model (all positions computed at once thanks to the causal mask) — here
    we do it step by step so each number is visible.
    """
    logprobs = []
    running = list(context_ids)
    for tok in continuation_ids:
        lp_vec = model.next_token_logprobs(running)   # distribution over ALL of vocab
        logprobs.append(float(lp_vec[tok]))            # pick out just this token
        running.append(tok)                             # then condition on it too
    return logprobs


def loglikelihood(model: ToyLM, context: str, continuation: str,
                   verbose: bool = False) -> float:
    """
    THE function that powers MMLU, HellaSwag, ARC (and any other
    multiple-choice benchmark). No sampling. No generation. We already know
    the exact string we're scoring — we just ask "how surprised would the
    model be to produce this exact continuation, token by token?" and sum
    the log-probabilities.

        score(continuation) = sum_i log P(c_i | context, c_1..c_{i-1})

    Higher (less negative) = more likely = better answer, in the model's eyes.
    """
    ctx_ids = tokenize(context)
    cont_ids = tokenize(continuation)
    logprobs = token_logprobs_for_continuation(model, ctx_ids, cont_ids)
    if verbose:
        toks = [ID2TOK[t] for t in cont_ids]
        for tok, lp in zip(toks, logprobs):
            print(f"    log P({tok!r} | ...) = {lp:.4f}   (P = {math.exp(lp):.4f})")
    return float(sum(logprobs))


def multiple_choice_score(model: ToyLM, context: str,
                           choices: List[str], verbose: bool = False
                           ) -> Tuple[int, List[float]]:
    """
    The MMLU inner loop: score every candidate answer against the same
    context, then argmax. This is `n_choices` separate loglikelihood calls
    — no text is ever generated.
    """
    scores = []
    for choice in choices:
        if verbose:
            print(f"  Scoring choice: {choice!r}")
        s = loglikelihood(model, context, choice, verbose=verbose)
        if verbose:
            print(f"    TOTAL log-likelihood = {s:.4f}\n")
        scores.append(s)
    best = int(np.argmax(scores))
    return best, scores


# ---------------------------------------------------------------------------
# 3. generate_until — how GSM8K (free-form generation) actually works
# ---------------------------------------------------------------------------

def generate_until(model: ToyLM, context: str, stop_strings: List[str],
                    max_new_tokens: int = 20, greedy: bool = True,
                    verbose: bool = False) -> str:
    """
    THE function that powers GSM8K-style tasks: the model doesn't know the
    right continuation in advance, so it must autoregressively PRODUCE one,
    token by token, feeding each new token back in as input to get the next,
    until it emits a stop string (e.g. "\\n\\n", "Question:") or hits a
    token budget.

    This is fundamentally different from loglikelihood(): there we scored a
    *given* string; here we are *searching* for one, one token at a time,
    and every one of those steps is its own forward pass through the model.
    """
    ids = tokenize(context)
    produced: List[int] = []
    text_so_far = context

    for step in range(max_new_tokens):
        lp_vec = model.next_token_logprobs(ids)
        if greedy:
            next_id = int(np.argmax(lp_vec))
        else:
            probs = np.exp(lp_vec)
            next_id = int(np.random.choice(V, p=probs))

        tok_str = ID2TOK[next_id]
        ids.append(next_id)
        produced.append(next_id)
        text_so_far += " " + tok_str

        if verbose:
            print(f"    step {step+1}: forward pass -> pick {tok_str!r} "
                  f"(logP={lp_vec[next_id]:.3f})  | text so far: {text_so_far!r}")

        if any(stop in text_so_far for stop in stop_strings):
            break

    return text_so_far


# ---------------------------------------------------------------------------
# 4. Memory-cost math: why loglikelihood fits an 8GB laptop and
#    unconstrained generate_until can OOM.
# ---------------------------------------------------------------------------

def kv_cache_bytes(n_layers: int, n_heads: int, head_dim: int,
                    seq_len: int, batch_size: int = 1,
                    dtype_bytes: int = 2) -> int:
    """
    KV-cache size formula (the actual memory driver during generation):

        bytes = 2 (K and V) * n_layers * n_heads * head_dim * seq_len
                * batch_size * dtype_bytes

    This is the memory a transformer must keep *in addition to* the model
    weights themselves, and it grows linearly with how many tokens have
    been generated/consumed so far (seq_len).
    """
    return 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"
