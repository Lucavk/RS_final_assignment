#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: tune → train → blend → submit
# Run from the project root:   bash run.sh
#
# Pick GPU 1 if GPU 0 is busy:  CUDA_VISIBLE_DEVICES=1 bash run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Make 'src' importable from any working directory; silence OpenBLAS warning
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1

MODELS="ease als itemknn popularity bpr multvae content"

echo "============================================================"
echo "  Recommender Systems Final Assignment — full pipeline"
echo "============================================================"

# ── 1. Tune each model on Fold B ─────────────────────────────────────────────
echo ""; echo "STAGE 1: Hyperparameter tuning (Fold B)"
echo "------------------------------------------------------------"
python src/tune.py --model ease       --fold b
python src/tune.py --model itemknn    --fold b
python src/tune.py --model als        --fold b
python src/tune.py --model bpr        --fold b
python src/tune.py --model multvae    --fold b
python src/tune.py --model content    --fold b
python src/tune.py --model popularity --fold b

# ── 2. Train all models, cache scores for BOTH folds ─────────────────────────
echo ""; echo "STAGE 2: Train + cache score matrices (Fold B for tuning, Fold A faithful)"
echo "------------------------------------------------------------"
python src/train_all.py --fold b --models $MODELS
python src/train_all.py --fold a --models $MODELS

# ── 3. Optimise ensemble weights (tune on B, check on A) ─────────────────────
echo ""; echo "STAGE 3: Optimise ensemble blend weights"
echo "------------------------------------------------------------"
python src/ensemble/blend.py --models $MODELS --tune_fold b --check_fold a --n_trials 300

# ── 4. Generate submission ───────────────────────────────────────────────────
echo ""; echo "STAGE 4: Generate submission CSV"
echo "------------------------------------------------------------"
python src/submit.py --models $MODELS

echo ""
echo "============================================================"
echo "  Done!  Submit:  data/submission.csv"
echo "============================================================"
