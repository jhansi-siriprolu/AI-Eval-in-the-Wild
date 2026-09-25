#!/usr/bin/env bash
# GSM8K requires generate_until (free-form generation) which OOMs on 8GB M1.
# Default tasks are the three log-likelihood benchmarks that fit comfortably.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python -c "import mlx_lm" 2>/dev/null; then
  if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
fi
if ! python -c "import mlx_lm" 2>/dev/null; then
  echo "mlx_lm not found. Run: pip install -r requirements.txt"; exit 1
fi

echo "Using: $(command -v python)"
python -c "import mlx_lm; print('mlx-lm version:', mlx_lm.__version__)"

MODEL="${1:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
TASKS="${2:-mmlu,hellaswag,arc_easy}"
LIMIT="${3:-50}"

echo "Model : $MODEL"
echo "Tasks : $TASKS"
echo "Limit : $LIMIT samples/task"
echo "(Note: GSM8K excluded -- it requires text generation and OOMs on 8GB M1)"
echo

python scripts/_run_eval.py --model "$MODEL" --tasks "$TASKS" --limit "$LIMIT" --out results/raw_eval_output.json

echo
echo "Done. Next: python scripts/05_export_results.py"
