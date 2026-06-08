"""
Ensemble blending: optimise per-model weights for RRF on Fold A.

Loads cached score matrices from artifacts/scores/  (produced by train_all.py)
and finds weights that maximise Recall@10 on the 2,255 submission-user targets.

Usage:
    python src/ensemble/blend.py --fold a
    python src/ensemble/blend.py --fold a --models ease als itemknn popularity

Writes the best weights to artifacts/params/ensemble_weights.json.
submit.py reads this file automatically.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.config import Config, RANDOM_SEED
from src.metrics import compute_metrics
from src.evaluate import load_scores

np.random.seed(RANDOM_SEED)


def rrf_blend_from_scores(score_matrices, weights, rrf_k=60):
    """Weighted RRF blend (same logic as submit.py)."""
    total = sum(weights)
    blended = np.zeros_like(score_matrices[0])
    for mat, w in zip(score_matrices, weights):
        order = np.argsort(-mat, axis=1)
        ranks = np.empty_like(order)
        rows  = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)
        blended += (w / total) / (rrf_k + ranks)
    return blended


def evaluate_weights(weights, score_matrices, target_item_idxs, user_seen_idxs):
    blended = rrf_blend_from_scores(score_matrices, weights)
    m = compute_metrics(blended, target_item_idxs, user_seen_idxs)
    return m["recall@10"]


def optimise_weights(
    model_names: list,
    fold: str = "a",
    n_trials: int = 200,
) -> dict:
    """Optuna coordinate-ascent over non-negative model weights."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    # Load cached scores
    all_scores = []
    for m in model_names:
        scores, targets, seen = load_scores(m, fold)
        all_scores.append(scores)
        # All meta must be consistent — use the first model's meta
        if m == model_names[0]:
            target_item_idxs = targets
            user_seen_idxs   = seen

    n = len(model_names)

    def objective(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in model_names]
        if sum(weights) < 1e-9:
            return 0.0
        return evaluate_weights(weights, all_scores, target_item_idxs, user_seen_idxs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    # Warm start: equal weights
    study.enqueue_trial({f"w_{m}": 1.0 for m in model_names})

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_weights = {m: study.best_params[f"w_{m}"] for m in model_names}
    best_score   = study.best_value

    print(f"\nBest ensemble  recall@10={best_score:.6f}")
    print(f"Weights: {best_weights}")

    # Baseline: equal weights
    eq_score = evaluate_weights(
        [1.0]*n, all_scores, target_item_idxs, user_seen_idxs
    )
    print(f"Equal weights  recall@10={eq_score:.6f}")
    print(f"Gain: +{(best_score-eq_score)*100:.3f} pp")

    # Save best weights
    Config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = Config.PARAMS_DIR / "ensemble_weights.json"
    with open(out, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"Saved → {out}")

    return best_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     default="a", choices=["a", "b"])
    parser.add_argument("--models",   nargs="+",   default=Config.ENSEMBLE_MODELS)
    parser.add_argument("--n_trials", type=int,    default=200)
    args = parser.parse_args()
    optimise_weights(args.models, fold=args.fold, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
