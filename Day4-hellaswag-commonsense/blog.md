# Can a Machine Really Finish Your Sentence?

### HellaSwag, the test that catches AI pretending to have common sense — and the cat-and-mouse trick that made it hard to fool

*AI Eval in the Wild · Day 4*

---

Read this and finish it:

> **A man is sitting on a roof. He…**

Which ending fits?

- **A.** …is using wrap to wrap a pair of skis.
- **B.** …is ripping level tiles off.
- **C.** …is holding a rubik's cube.
- **D.** …starts pulling up roofing on a roof.

You answered in about one second. You did not look anything up. You just *knew* how roofs, men and shingles behave.

Now here is the uncomfortable part: for a computer, this one-second task is a genuine challenge. People get about **95.6%** of these questions right. Someone guessing at random gets **25%**. The test is called **HellaSwag**, and today I want to show you how it works, how it was deliberately built to be *unfair to machines*, and what a score like "55% on a 1.5 billion parameter model" really tells you.

Everything below is backed by code I actually ran. Numbers come from a real run on a real model, not from my imagination.

---

## The name is a mouthful. The idea is not.

HellaSwag stands for *Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations*. It is the grown-up sibling of an older test called SWAG.

Each question has four pieces:

1. An **activity label** ("Roof shingle removal").
2. A **context** — the opening of a description, taken from video captions or how-to articles.
3. **Four endings** — one is what really happened next, three are impostors written by a computer.
4. A **label** saying which ending is the real one.

The dataset has 39,905 practice questions, 10,042 exam questions with answers, and 10,003 exam questions whose answers are kept secret.

Look again at our roof question. Skis and a rubik's cube are perfectly grammatical English. "Ripping level tiles off" is *almost* right — but nobody says "level tiles" about a roof. The impostors are not stupid. That is the entire point.

---

## The puzzle: a next-word guesser that must answer multiple choice

A language model like GPT-2 (*Generative Pre-trained Transformer 2*) knows exactly one trick: given some text, guess what comes next. It has never seen the letters A, B, C, D.

So how do you make it take a multiple-choice exam?

**You ask the same question four times: "How likely is it that this ending follows this context?" Then you pick the most likely one.**

Here is the maths, in plain words.

**Step one — probabilities.** For every next chunk of text (called a *token*), the model gives a probability between 0 (impossible) and 1 (certain). After "She flipped the", the token "pancake" gets a high probability and "garage" a tiny one. Probabilities that depend on what came before are called *conditional probabilities*.

**Step two — the chain rule.** The probability of a whole ending is every token's probability multiplied together.

**Step three — the underflow problem.** Multiply lots of small numbers and the result gets absurdly tiny. I tested it: multiply 300 probabilities of 0.001 and the computer answers *exactly zero*. It ran out of room to store the number.

**Step four — logarithms to the rescue.** The natural logarithm turns multiplication into addition. The natural logarithm of 1 is 0, of 0.5 is about −0.69, of 0.01 is about −4.6. Smaller probability means a more negative number. So we add up the logarithm of each token's probability, and the ending closest to zero wins.

Same 300 probabilities, added as logarithms: −2072.3. A perfectly healthy number.

---

## The plot twist: longer endings get punished for being longer

Every extra token adds another negative number to the sum. So plain adding **favours short endings**, whether or not they are right.

Picture two endings. Ending A has 3 tokens, each with probability 0.1. Ending B has 6 tokens, each with probability 0.2. B is clearly made of *better* tokens. Yet:

- Total for A: −6.91
- Total for B: −9.66

A wins the sum. B wins per token (−1.61 versus −2.30). Two scoring rules, two different winners.

There are three rules in common use: the plain **total**, the **average per token**, and the **score per character** (which the popular evaluation harness calls "length-normalised accuracy").

Does it matter in practice? Here is the real roof question, scored by real GPT-2 (the small, 124-million-parameter one):

- Add up log probabilities → picks **the rubik's cube**. Wrong.
- Average per token → picks **the rubik's cube**. Wrong.
- Score per character → picks **pulling up roofing**. Correct!

Same model. Same question. Three rules, two different answers. **A HellaSwag score means nothing until you know which scoring rule produced it.**

---

## Scoring 1,000 questions

One question proves nothing. So I scored 1,000 random exam questions:

- Guessing: **25.0%**
- GPT-2 small, adding up: **26.6%**
- GPT-2 small, average per token: **30.4%**
- GPT-2 small, per character: **30.3%**
- Humans: **95.6%**

Adding up is clearly the weakest, and the two fixed rules land almost on top of each other. The margin of error on 1,000 questions is about ±2.8 points, so tiny gaps are noise. (The formula: 1.96 × the square root of accuracy × (1 − accuracy) ÷ number of questions.)

---

## Can you cheat without understanding anything?

Here is what I love about HellaSwag: the authors *tried to make cheating impossible*. So I tried to cheat, using three tricks that need zero common sense:

1. **Pick the longest ending.**
2. **Pick the ending sharing the most words with the context.** Roof in the context, roof in the ending — looks right!
3. **Train a small classifier on 4,000 practice questions** to spot "real-sounding" endings *from the ending alone*, using single words and two-word runs.

To see what the test's design achieved, I also built an *easy* exam where the wrong endings are borrowed from random other questions (a roof question might get a baking ending). Results:

- Longest ending: 25.7% on the easy exam, 22.4% on real HellaSwag.
- Word overlap: **66.5%** on the easy exam, **34.2%** on real HellaSwag.
- Ending-only classifier: **31.3%** on real HellaSwag.

Word overlap is a champion on the easy exam, because topic words give the game away — and it falls off a cliff on the real one. The wrong endings were chosen to talk about the same topic. The shortcut has nothing to grab.

How? That brings us to the best idea in the whole benchmark.

---

## The forger and the detective

The trick is called **Adversarial Filtering**, and it is a game with two players:

- A **generator** — a text-writing model — churns out a huge pool of fake endings.
- A **discriminator** — a classifier — learns to tell real endings from fake ones.

Keep the fakes that **fool the detective**. Throw away the obvious ones. Retrain the detective on the tougher pile. Repeat.

Imagine a forger and a detective sharpening each other until the detective is barely better than a coin toss.

I built a miniature version. To be honest about it: my generator is a simple bigram Markov chain (it picks each next word looking only at the previous word), so it is a toy next to what the real dataset used. But the mechanism is identical, and the result is striking. A detective trying to spot the real ending among four:

- Before any filtering: **62.5%** correct
- After one round: **31.0%**
- After six rounds: **28.1%** — practically guessing

Notice this is exactly why the cheap tricks in the previous section fail. The impostors now look real to *anything that reads surface patterns*.

---

## So what does "55% on 1.5 billion parameters" mean?

A *parameter* is one adjustable dial inside the model, so "1.5 billion parameters" means 1.5 billion dials. Let's decode the headline:

- **Distance travelled.** Guessing is 25%, humans are 95.6%. A score of 55% closes (55 − 25) ÷ (95.6 − 25) = about **42%** of that gap.
- **Mistakes.** 55% means **45 wrong answers in every 100**.
- **Wobble.** On all 10,042 exam questions, the margin of error is about ±1.0 point. A 55% model and a 56% model are practically tied.
- **Size context.** Roughly speaking, the small 124-million-parameter GPT-2 lands near 30% (my run agrees), the 1.5-billion-parameter one near 50%, and today's biggest models above 90%. So 55% at 1.5 billion is respectable and unspectacular: real knowledge about everyday scenes, still wrong nearly half the time.

(The human figure comes from the HellaSwag paper; the model-size figures are approximate and worth re-checking with a proper evaluation run.)

---

## Why is common sense harder than facts?

Ask a model for the capital of France and it nails it. Ask what happens after a man loosens a shingle and it stumbles. Why?

**Because facts get written down and common sense does not.** The internet states "Paris is the capital of France" thousands of times. Nobody writes "a wet towel is heavier than a dry one." Researchers call this **reporting bias**: people write about the remarkable, not the obvious. A model that learns from text only meets the obvious sideways.

**Because facts are lookups and common sense is a simulation.** "Capital of France" points at one answer. "What happens next?" needs an imagined mini-world with objects, gravity and human habits.

**Because facts can be matched, but common sense is judged by plausibility.** Several wrong endings can be grammatical *and* on topic. Only one fits the scene.

You can see it in the model's mistakes. Given "Washing hands: a man in scrubs stands in front of a sink. The man…", small GPT-2 chose "…then gets down on his knee and begins washing dishes." Fluent. Story-like. And it ignores the label that literally says *washing hands*.

---

## What HellaSwag still misses

Please do not walk away thinking a high score equals a mind that understands the world. HellaSwag has real limits:

1. **A narrow slice of common sense.** Mostly everyday physical activities. Social reasoning, long chains of cause and effect, numbers, time and "what if" questions are barely touched.
2. **Choosing is not writing.** A model that ranks four endings may still write nonsense. The test never asks it to write.
3. **Machine fingerprints.** The impostors came from specific generators. A model might learn their style rather than the scene.
4. **Noisy answers.** Some questions have two defensible endings.
5. **Saturation.** The best models are near human level, so the test can no longer separate the best from the very best.
6. **Possible contamination.** The test has been public since 2019. Text resembling it may sit inside training data.
7. **Only English, and only two text sources.** In my run, small GPT-2 scored 39.5% on video-caption questions but only **25.7%** on wikiHow articles — no better than guessing.
8. **A score is not an explanation.** Passing says the model picked well, not *why*.

---

## The one-minute version

- HellaSwag: pick the true ending out of four. Guessing scores 25%, humans about 95.6%.
- A next-word model answers by comparing each ending's probability, using **sums of logarithms** to stay numerically safe and **length normalisation** to stay fair.
- The wrong endings are machine-written and **adversarially filtered**, so word-counting shortcuts fail.
- 55% at 1.5 billion parameters is real skill, far from human, and only meaningful once you know the scoring rule.
- Common sense is hard because it is *unwritten*.

---

## Try it yourself

Everything in this post is reproducible. The lesson ships as an interactive web page, a notebook and a single Python script that follow the same eight sections:

```
python hellaswag_demo.py                  # small model, 1,000 questions
python hellaswag_demo.py --model gpt2-xl  # the 1.5 billion parameter model
```

Change the settings, rerun, and see whether your numbers agree with mine. That is the whole spirit of this series: nothing taken on faith.

*Next up in AI Eval in the Wild: coming soon.*

---

**Source:** Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?* (2019). Dataset: `Rowan/hellaswag` on Hugging Face.
