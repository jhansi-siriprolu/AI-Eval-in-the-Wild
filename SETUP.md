# Setup

## Prerequisites

- **macOS 13.5+** on **Apple Silicon** (your M1 Air qualifies).
- **Xcode Command Line Tools** — install with `xcode-select --install` if you
  haven't already (needed to compile some Python packages).
- **Python 3.9+** — macOS ships one at `/usr/bin/python3`; that's fine.

Confirm you're on Apple Silicon before going further:

```bash
python3 -c "import platform; print(platform.machine())"
# should print: arm64
```

## 1. Create and activate a virtual environment

```bash
cd ~/Documents/"AI Evals"
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Every command below assumes
this is active — if you close Terminal and come back later, re-run the
`source .venv/bin/activate` line first.

## 2. Install everything

Either run the helper script:

```bash
bash scripts/01_setup_env.sh
```

or do it by hand:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `mlx` | Apple's array/ML framework, runs on the M1's GPU via Metal |
| `mlx-lm` | Language model utilities on top of MLX (download/generate/chat/evaluate) |
| `huggingface_hub` | Download model repos from the Hugging Face Hub |
| `lm-eval` | The standard benchmark harness (MMLU, HellaSwag, ARC, GSM8K, ...) |

## 3. Verify it works

```bash
python scripts/02_download_and_test.py
```

This downloads `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~1GB, one-time)
and runs one test prompt. If you see generated text and a tokens/sec figure
at the end, you're set — head to `lessons/index.html` and start with
Lesson 01, or jump to Lesson 05 if you just want the play-by-play of what
you just ran.

## Common issues

| Problem | Fix |
|---|---|
| `pip install mlx` fails | Confirm `arm64` from the prerequisite check above — MLX does not run on Intel Macs |
| Very slow / swapping | Close other memory-heavy apps; check Activity Monitor's Memory tab while the model runs |
| `xcrun: error: invalid active developer path` | Run `xcode-select --install` |
| Model download seems to hang | It's ~1GB on first run — check Activity Monitor's Network tab; it's usually just slow, not stuck |
