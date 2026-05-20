#!/usr/bin/env bash
# run_ablation_evals.sh
# Waits for the no-PBRS training process to exit, then runs both evaluation
# scripts against the final checkpoint, writing results to results_no_pbrs/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="models_no_pbrs/ppo_footsies_final.zip"
RESULTS_DIR="results_no_pbrs"

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

# ── Wait for training to finish ───────────────────────────────────────────────
echo "[ablation-evals] Waiting for train.py --no-pbrs to finish..."
while pgrep -f "train.py.*--no-pbrs" > /dev/null 2>&1; do
    sleep 30
done
echo "[ablation-evals] Training process gone. Checking for final model..."

# Give SB3 a moment to flush the .zip write
sleep 5

if [[ ! -f "$MODEL" ]]; then
    echo "[ablation-evals] ERROR: $MODEL not found. Training may have crashed." >&2
    exit 1
fi
echo "[ablation-evals] Found $MODEL — starting evaluations."

# ── League eval ───────────────────────────────────────────────────────────────
echo ""
echo "=== League evaluation ==="
python eval_league.py \
    --model "$MODEL" \
    --matches 100 \
    --speed 20 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/league_eval.log"

# ── Scenario eval ─────────────────────────────────────────────────────────────
echo ""
echo "=== Scenario evaluation ==="
python eval_scenarios.py \
    --model "$MODEL" \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/scenario_eval.log"

echo ""
echo "[ablation-evals] Done."
echo "  League results : $RESULTS_DIR/league_results.csv"
echo "  League summary : $RESULTS_DIR/league_summary.json"
echo "  Scenario CSV   : $RESULTS_DIR/scenario_results.csv"
