#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: tune → train → blend → submit
# Run from the project root:
#   bash run.sh
#
# Stages (uncomment/skip as needed):
#   1. Tune each Tier-1 model on Fold B
#   2. Train all models on Fold A with best params → cache scores
#   3. Optimise ensemble weights on Fold A
#   4. Generate the final submission CSV
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Recommender Systems Final Assignment — full pipeline"
echo "============================================================"

# ── 1. Tune on Fold B ────────────────────────────────────────────────────────
echo ""
echo "STAGE 1: Hyperparameter tuning (Fold B)"
echo "------------------------------------------------------------"

python src/tune.py --model ease       --fold b --n_trials 15
python src/tune.py --model itemknn    --fold b --n_trials 40
python src/tune.py --model als        --fold b --n_trials 50
python src/tune.py --model popularity --fold b --n_trials 10

# ── 2. Train all models on Fold A, cache score matrices ──────────────────────
echo ""
echo "STAGE 2: Train on Fold A → cache score matrices"
echo "------------------------------------------------------------"

python src/train_all.py --fold a

# Also cache Fold B scores (needed for report / ensemble analysis)
# python src/train_all.py --fold b   # optional, slower

# ── 3. Optimise ensemble weights ─────────────────────────────────────────────
echo ""
echo "STAGE 3: Optimise ensemble blend weights (Fold A)"
echo "------------------------------------------------------------"

python src/ensemble/blend.py --fold a --n_trials 200

# ── 4. Generate submission ───────────────────────────────────────────────────
echo ""
echo "STAGE 4: Generate submission CSV"
echo "------------------------------------------------------------"

python src/submit.py

echo ""
echo "============================================================"
echo "  Done!  Submit:  data/submission.csv"
echo "============================================================"
