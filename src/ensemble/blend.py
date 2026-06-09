"""
Ensemble blending: optimise per-model RRF weights.

Reciprocal Rank Fusion (RRF): each model contributes 1/(k + rank) per item;
models are combined with non-negative weights.  Rank contributions are
PRECOMPUTED once per model, so each Optuna trial is just a cheap weighted sum
— this lets us tune on the large Fold B (low variance) instead of overfitting
the small Fold A.

Workflow:
    python src/train_all.py --fold b        # cache Fold B scores (tuning set)
    python src/train_all.py --fold a         # cache Fold A scores (faithful check)
    python src/ensemble/blend.py             # tune on B, report on A

Writes best weights to artifacts/params/ensemble_weights.json (read by submit.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import Config, RANDOM_SEED
from src.evaluate import load_scores

np.random.seed(RANDOM_SEED)


# ── Fast RRF utilities ────────────────────────────────────────────────────────

def precompute_rrf(score_matrices: list[np.ndarray], rrf_k: int = 60) -> list[np.ndarray]:
    """Convert each model's score matrix to its RRF rank-contribution matrix."""
    rrf_list = []
    for mat in score_matrices:
        order = np.argsort(-mat, axis=1)                  # item idx by descending score
        ranks = np.empty_like(order)
        rows  = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)
        rrf_list.append((1.0 / (rrf_k + ranks)).astype(np.float32))
    return rrf_list


def fast_recall(blended, target_item_idxs, user_seen_idxs, k=10) -> float:
    """Recall@k with seen-item masking (no NDCG → faster than compute_metrics)."""
    scores = blended.copy()
    for row, seen in enumerate(user_seen_idxs):
        if seen:
            scores[row, list(seen)] = -np.inf
    top_k = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = sum(target_item_idxs[i] in top_k[i] for i in range(len(target_item_idxs)))
    return hits / len(target_item_idxs) if target_item_idxs else 0.0


def blend_weighted(rrf_list, weights):
    """Weighted sum of precomputed RRF contributions."""
    total = sum(weights) or 1.0
    out = np.zeros_like(rrf_list[0])
    for rrf, w in zip(rrf_list, weights):
        if w:
            out += (w / total) * rrf
    return out


# ── Tuning ─────────────────────────────────────────────────────────────────────

def optimise_weights(model_names, tune_fold="b", check_fold="a",
                     n_trials=300, max_eval_users=10000):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Load tuning-fold scores ──
    print(f"Loading Fold {tune_fold.upper()} scores for {model_names}…")
    tune_scores, targets, seen = [], None, None
    for m in model_names:
        s, t, se = load_scores(m, tune_fold)
        tune_scores.append(s)
        if targets is None:
            targets, seen = t, se

    # Subsample rows (consistent across models) to bound memory/time
    n_eval = tune_scores[0].shape[0]
    if n_eval > max_eval_users:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = rng.choice(n_eval, max_eval_users, replace=False)
        tune_scores = [s[sel] for s in tune_scores]
        targets     = [targets[i] for i in sel]
        seen        = [seen[i] for i in sel]
        print(f"  subsampled to {max_eval_users:,} eval users")

    rrf_tune = precompute_rrf(tune_scores, Config.ENSEMBLE_RRF_K)

    def objective(trial):
        w = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in model_names]
        if sum(w) < 1e-9:
            return 0.0
        return fast_recall(blend_weighted(rrf_tune, w), targets, seen)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({f"w_{m}": 1.0 for m in model_names})   # warm-start: equal
    print(f"Tuning {len(model_names)} weights over {n_trials} trials on Fold {tune_fold.upper()}…")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_weights = {m: study.best_params[f"w_{m}"] for m in model_names}
    eq_score = fast_recall(blend_weighted(rrf_tune, [1.0]*len(model_names)), targets, seen)

    print(f"\n── Fold {tune_fold.upper()} (tuning) ──")
    print(f"  equal weights : recall@10={eq_score:.6f}")
    print(f"  tuned weights : recall@10={study.best_value:.6f}  (+{(study.best_value-eq_score)*100:.3f} pp)")
    print(f"  weights: { {m: round(w,3) for m,w in best_weights.items()} }")

    # ── Sanity-check on the faithful fold ──
    try:
        check_scores, c_targets, c_seen = [], None, None
        for m in model_names:
            s, t, se = load_scores(m, check_fold)
            check_scores.append(s)
            if c_targets is None:
                c_targets, c_seen = t, se
        rrf_check = precompute_rrf(check_scores, Config.ENSEMBLE_RRF_K)
        eq_a   = fast_recall(blend_weighted(rrf_check, [1.0]*len(model_names)), c_targets, c_seen)
        best_a = fast_recall(blend_weighted(rrf_check, [best_weights[m] for m in model_names]),
                             c_targets, c_seen)
        # Best single model on the faithful fold
        singles = {m: fast_recall(rrf_check[i], c_targets, c_seen)
                   for i, m in enumerate(model_names)}
        best_single = max(singles, key=singles.get)
        print(f"\n── Fold {check_fold.upper()} (faithful check) ──")
        for m in model_names:
            print(f"  {m:<12} recall@10={singles[m]:.6f}")
        print(f"  equal-weight ensemble : recall@10={eq_a:.6f}")
        print(f"  tuned-weight ensemble : recall@10={best_a:.6f}")
        print(f"  best single ({best_single}) : recall@10={singles[best_single]:.6f}")
    except FileNotFoundError:
        print(f"\n(no Fold {check_fold.upper()} scores cached — skipping faithful check)")

    # ── Save ──
    Config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = Config.PARAMS_DIR / "ensemble_weights.json"
    with open(out, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"\nSaved → {out}")
    return best_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=Config.ENSEMBLE_MODELS)
    parser.add_argument("--tune_fold",  default=Config.ENSEMBLE_TUNE_FOLD, choices=["a", "b"])
    parser.add_argument("--check_fold", default="a", choices=["a", "b"])
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--max_eval_users", type=int, default=10000)
    args = parser.parse_args()
    optimise_weights(args.models, tune_fold=args.tune_fold, check_fold=args.check_fold,
                     n_trials=args.n_trials, max_eval_users=args.max_eval_users)


if __name__ == "__main__":
    main()
