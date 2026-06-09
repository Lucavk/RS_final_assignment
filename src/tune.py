"""
Optuna hyperparameter tuning for Tier-1 models.

Each model gets its own Optuna study, persisted to artifacts/optuna/<model>.db.
Studies are resumable — re-running with the same model name adds more trials.

Usage:
    python src/tune.py --model ease   --fold b --n_trials 15
    python src/tune.py --model als    --fold b --n_trials 50
    python src/tune.py --model itemknn --fold b --n_trials 40
    python src/tune.py --model popularity --fold b --n_trials 10

    # After tuning all models, re-evaluate the best params on Fold A:
    python src/tune.py --model ease --fold a --n_trials 0   # just evaluates

Best params are saved to artifacts/params/<model>_best.json.
Then run  python src/train_all.py --fold a  to cache the final score matrices.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config, RANDOM_SEED
from src.data import build_bundle, build_id_maps, load_all_data, load_train_only, load_submission_user_ids
from src.splits import fold_a, fold_b, val_targets_to_arrays
from src.metrics import compute_metrics

np.random.seed(RANDOM_SEED)


# ── Hyperparameter search spaces ─────────────────────────────────────────────

def suggest_params(trial, model_name: str) -> dict:
    if model_name == "ease":
        return {"lam": trial.suggest_float("lam", 50.0, 2000.0, log=True)}

    if model_name == "itemknn":
        return {
            "topk":      trial.suggest_int("topk", 50, 500),
            "shrinkage": trial.suggest_float("shrinkage", 0.0, 500.0),
        }

    if model_name == "als":
        return {
            "factors":        trial.suggest_int("factors", 64, 256, step=64),
            "regularization": trial.suggest_float("regularization", 1e-3, 0.2, log=True),
            "alpha":          trial.suggest_float("alpha", 1.0, 100.0, log=True),
            "iterations":     trial.suggest_int("iterations", 15, 60),
        }

    if model_name == "popularity":
        return {
            "halflife_days": trial.suggest_float("halflife_days", 30.0, 3650.0, log=True),
        }

    raise ValueError(f"Unknown model: {model_name}")


# ── Objective factory ─────────────────────────────────────────────────────────

def make_objective(model_name, train_bundle, eval_user_idxs, target_item_idxs,
                   user_seen_idxs, train_df):
    """Return an Optuna objective that fits the model and returns Recall@10."""

    def objective(trial):
        params = suggest_params(trial, model_name)

        from src.train_all import build_model
        model = build_model(model_name, params)

        if model_name == "popularity" and model.halflife_days is not None:
            model.fit_with_decay(train_bundle, train_df)
        else:
            model.fit(train_bundle)

        scores  = model.score_users(eval_user_idxs)
        metrics = compute_metrics(scores, target_item_idxs, user_seen_idxs)
        return metrics["recall@10"]

    return objective


# ── Main tuning driver ────────────────────────────────────────────────────────

def tune(model_name: str, fold_name: str, n_trials: int, timeout: int | None = None):
    try:
        import optuna
    except ImportError:
        raise ImportError("Install optuna:  pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nTuning {model_name.upper()}  (fold={fold_name}, trials={n_trials})")
    print("Loading train.csv only (honest validation)…")
    df_full = load_train_only(Config)  # no time-leak from test.csv
    sub_ids = load_submission_user_ids(Config)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(df_full)

    if fold_name == "a":
        train_df, val_targets = fold_a(df_full, sub_ids)
    else:
        train_df, val_targets = fold_b(df_full)
        # Subsample to 5k users for faster Fold B tuning
        rng = np.random.default_rng(RANDOM_SEED)
        user_list = list(val_targets.keys())
        if len(user_list) > 5000:
            selected = rng.choice(len(user_list), 5000, replace=False)
            user_list = [user_list[i] for i in selected]
            val_targets = {u: val_targets[u] for u in user_list}
            print(f"  (subsampled fold B to {len(val_targets):,} users for tuning)")

    train_bundle = build_bundle(
        train_df, user_to_idx, idx_to_user, item_to_idx, idx_to_item, sub_ids
    )
    eval_user_idxs, target_item_idxs = val_targets_to_arrays(
        val_targets, user_to_idx, item_to_idx
    )
    user_seen_idxs = [
        train_bundle.user_seen_idxs.get(u, set()) for u in eval_user_idxs
    ]

    print(f"  {len(eval_user_idxs):,} eval users | "
          f"{train_bundle.n_items:,} items")

    # ── Optuna study ──────────────────────────────────────────────────────────
    Config.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = Config.OPTUNA_DIR / f"{model_name}.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=model_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    objective = make_objective(
        model_name, train_bundle, eval_user_idxs,
        target_item_idxs, user_seen_idxs, train_df
    )

    if n_trials > 0:
        print(f"  Running {n_trials} Optuna trials…  (db: {db_path})")
        t0 = time.time()
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )
        print(f"  Done in {time.time()-t0:.0f}s")

    best = study.best_trial
    print(f"\nBest  recall@10={best.value:.6f}")
    print(f"Best params: {best.params}")

    # Save best params
    Config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    params_path = Config.PARAMS_DIR / f"{model_name}_best.json"
    with open(params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved → {params_path}")

    # Print top-5 trials
    all_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    print("\nTop-5 trials:")
    for t in all_trials[:5]:
        print(f"  trial #{t.number:3d}  recall@10={t.value:.6f}  {t.params}")

    return best.params, best.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["popularity", "itemknn", "ease", "als"])
    parser.add_argument("--fold",    default="b",    choices=["a", "b"])
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Overrides Config.TUNE_N_TRIALS if provided")
    parser.add_argument("--timeout",  type=int, default=None,
                        help="Optuna wall-time limit in seconds")
    args = parser.parse_args()

    n_trials = args.n_trials if args.n_trials is not None \
               else Config.TUNE_N_TRIALS.get(args.model, 20)
    timeout  = args.timeout  if args.timeout  is not None \
               else Config.TUNE_TIMEOUT.get(args.model, None)

    tune(args.model, args.fold, n_trials, timeout)


if __name__ == "__main__":
    main()
