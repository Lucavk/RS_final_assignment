from __future__ import annotations
from src.evaluate import load_scores
from src.config import Config, RANDOM_SEED
import numpy as np

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


np.random.seed(RANDOM_SEED)


# Helpers for rank based blending.

def precompute_rrf(score_matrices: list[np.ndarray], rrf_k: int = 60) -> list[np.ndarray]:
    # Turn model scores into rank scores once, so tuning is faster.
    rrf_list = []
    for mat in score_matrices:
        # item idx by descending score
        order = np.argsort(-mat, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)
        rrf_list.append((1.0 / (rrf_k + ranks)).astype(np.float32))
    return rrf_list


def fast_recall(blended, target_item_idxs, user_seen_idxs, k=10) -> float:
    # Quick recall check while trying many weight settings.
    scores = blended.copy()
    for row, seen in enumerate(user_seen_idxs):
        if seen:
            scores[row, list(seen)] = -np.inf
    top_k = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = sum(target_item_idxs[i] in top_k[i]
               for i in range(len(target_item_idxs)))
    return hits / len(target_item_idxs) if target_item_idxs else 0.0


def blend_weighted(rrf_list, weights):
    # Add the model scores together using the given weights.
    total = sum(weights) or 1.0
    out = np.zeros_like(rrf_list[0])
    for rrf, w in zip(rrf_list, weights):
        if w:
            out += (w / total) * rrf
    return out


# Try different weights and keep the best set.
def optimise_weights(model_names, tune_fold="b", check_fold="a",
                     n_trials=300, max_eval_users=10000):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load the saved scores for the fold used to tune the weights.
    print(f"Loading Fold {tune_fold.upper()} scores for {model_names}…")
    tune_scores, targets, seen = [], None, None
    for m in model_names:
        s, t, se = load_scores(m, tune_fold)
        tune_scores.append(s)
        if targets is None:
            targets, seen = t, se

    # Use a smaller random group if the fold is too large.
    n_eval = tune_scores[0].shape[0]
    if n_eval > max_eval_users:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = rng.choice(n_eval, max_eval_users, replace=False)
        tune_scores = [s[sel] for s in tune_scores]
        targets = [targets[i] for i in sel]
        seen = [seen[i] for i in sel]
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
    # Start with equal weights.
    study.enqueue_trial({f"w_{m}": 1.0 for m in model_names})
    print(
        f"Tuning {len(model_names)} weights over {n_trials} trials on Fold {tune_fold.upper()}…")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_weights = {m: study.best_params[f"w_{m}"] for m in model_names}
    eq_score = fast_recall(blend_weighted(
        rrf_tune, [1.0]*len(model_names)), targets, seen)

    print(f"\nFold {tune_fold.upper()} tuning")
    print(f"  equal weights : recall@10={eq_score:.6f}")
    print(
        f"  tuned weights : recall@10={study.best_value:.6f}  (+{(study.best_value-eq_score)*100:.3f} pp)")
    print(f"  weights: { {m: round(w,3) for m,w in best_weights.items()} }")

    # Check the weights on the other fold as a simple safety check.
    try:
        check_scores, c_targets, c_seen = [], None, None
        for m in model_names:
            s, t, se = load_scores(m, check_fold)
            check_scores.append(s)
            if c_targets is None:
                c_targets, c_seen = t, se
        rrf_check = precompute_rrf(check_scores, Config.ENSEMBLE_RRF_K)
        eq_a = fast_recall(blend_weighted(
            rrf_check, [1.0]*len(model_names)), c_targets, c_seen)
        best_a = fast_recall(blend_weighted(rrf_check, [best_weights[m] for m in model_names]),
                             c_targets, c_seen)
        # Also show the best single model for comparison.
        singles = {m: fast_recall(rrf_check[i], c_targets, c_seen)
                   for i, m in enumerate(model_names)}
        best_single = max(singles, key=singles.get)
        print(f"\nFold {check_fold.upper()} check")
        for m in model_names:
            print(f"  {m:<12} recall@10={singles[m]:.6f}")
        print(f"  equal-weight ensemble : recall@10={eq_a:.6f}")
        print(f"  tuned-weight ensemble : recall@10={best_a:.6f}")
        print(
            f"  best single ({best_single}) : recall@10={singles[best_single]:.6f}")
    except FileNotFoundError:
        print(
            f"\n(no Fold {check_fold.upper()} scores cached — skipping faithful check)")

    # Save the weights so submit.py can use them.
    Config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = Config.PARAMS_DIR / "ensemble_weights.json"
    with open(out, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"\nSaved → {out}")
    return best_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+",
                        default=Config.ENSEMBLE_MODELS)
    parser.add_argument(
        "--tune_fold",  default=Config.ENSEMBLE_TUNE_FOLD, choices=["a", "b"])
    parser.add_argument("--check_fold", default="a", choices=["a", "b"])
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--max_eval_users", type=int, default=10000)
    args = parser.parse_args()
    optimise_weights(args.models, tune_fold=args.tune_fold, check_fold=args.check_fold,
                     n_trials=args.n_trials, max_eval_users=args.max_eval_users)


if __name__ == "__main__":
    main()
