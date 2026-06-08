"""
Train all Tier-1 models on a given fold and cache their score matrices.

Usage:
    python src/train_all.py --fold a          # Fold A (submission LOO) — default
    python src/train_all.py --fold b          # Fold B (global LOO)
    python src/train_all.py --fold a --models ease als   # subset of models

Outputs (per model):
    artifacts/scores/<model>_<fold>.npy        — float32 [n_eval × n_items]
    artifacts/scores/<model>_<fold>_meta.pkl   — targets + seen-item sets

Then run:
    python src/evaluate.py --fold a
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.config import Config, RANDOM_SEED
from src.data import (
    DataBundle,
    build_bundle,
    build_id_maps,
    load_all_data,
    load_submission_user_ids,
)
from src.splits import fold_a, fold_b, val_targets_to_arrays
from src.metrics import compute_metrics
from src.evaluate import print_leaderboard

np.random.seed(RANDOM_SEED)


# ── Score caching helpers ────────────────────────────────────────────────────

def save_scores(
    model_name: str,
    fold: str,
    score_matrix: np.ndarray,
    target_item_idxs: list,
    user_seen_idxs: list,
    scores_dir: Path,
) -> None:
    scores_dir.mkdir(parents=True, exist_ok=True)
    np.save(scores_dir / f"{model_name}_{fold}.npy", score_matrix)
    meta = {"target_item_idxs": target_item_idxs, "user_seen_idxs": user_seen_idxs}
    with open(scores_dir / f"{model_name}_{fold}_meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=4)


def load_best_params(model_name: str) -> dict:
    """Load tuned hyperparams if they exist, otherwise return empty dict."""
    path = Config.PARAMS_DIR / f"{model_name}_best.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"  [params] loaded tuned params for {model_name}: {params}")
        return params
    return {}


# ── Model factory ────────────────────────────────────────────────────────────

def build_model(model_name: str, params: dict | None = None):
    """Instantiate the model, applying tuned params where available."""
    p = params or load_best_params(model_name)

    if model_name == "popularity":
        from src.models.popularity import PopularityRecommender
        return PopularityRecommender(
            halflife_days=p.get("halflife_days", Config.POPULARITY_HALFLIFE_DAYS)
        )
    if model_name == "itemknn":
        from src.models.itemknn import ItemKNNRecommender
        return ItemKNNRecommender(
            topk=p.get("topk", Config.ITEMKNN_TOPK),
            shrinkage=p.get("shrinkage", Config.ITEMKNN_SHRINKAGE),
        )
    if model_name == "ease":
        from src.models.ease import EASERecommender
        return EASERecommender(lam=p.get("lam", Config.EASE_LAMBDA))

    if model_name == "als":
        from src.models.als import ALSRecommender
        return ALSRecommender(
            factors=p.get("factors", Config.ALS_FACTORS),
            regularization=p.get("regularization", Config.ALS_REGULARIZATION),
            iterations=p.get("iterations", Config.ALS_ITERATIONS),
            alpha=p.get("alpha", Config.ALS_ALPHA),
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ── Main training loop ───────────────────────────────────────────────────────

ALL_MODELS = ["popularity", "itemknn", "ease", "als"]


def run(fold_name: str = "a", model_names: list | None = None, df_full=None) -> dict:
    """
    Train each model on the given fold, cache score matrices, return metrics.
    """
    if model_names is None:
        model_names = ALL_MODELS

    # ── Data ──────────────────────────────────────────────────────────────────
    if df_full is None:
        print("Loading data…")
        df_full = load_all_data(Config)

    sub_ids = load_submission_user_ids(Config)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(df_full)

    print(f"\nBuilding Fold {fold_name.upper()}…")
    if fold_name == "a":
        train_df, val_targets = fold_a(df_full, sub_ids)
    elif fold_name == "b":
        train_df, val_targets = fold_b(df_full)
    else:
        raise ValueError(f"Unknown fold: {fold_name}")

    print(f"  train interactions : {len(train_df):,}")
    print(f"  eval users         : {len(val_targets):,}")

    train_bundle = build_bundle(
        train_df, user_to_idx, idx_to_user, item_to_idx, idx_to_item, sub_ids
    )

    # Align val_targets with matrix indices
    eval_user_idxs, target_item_idxs = val_targets_to_arrays(
        val_targets, user_to_idx, item_to_idx
    )

    # seen-item sets for evaluation (use train-only history)
    user_seen_idxs = [
        train_bundle.user_seen_idxs.get(u_idx, set())
        for u_idx in eval_user_idxs
    ]

    # ── Train + score each model ──────────────────────────────────────────────
    results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training  {model_name.upper()}  (fold={fold_name})…")
        print(f"{'='*60}")

        t_start = time.time()
        model = build_model(model_name)

        # Popularity can use time-decay if we pass df
        if model_name == "popularity" and model.halflife_days is not None:
            model.fit_with_decay(train_bundle, train_df)
        else:
            model.fit(train_bundle)

        t_fit = time.time() - t_start
        print(f"  fit time: {t_fit:.1f}s")

        print(f"  scoring {len(eval_user_idxs):,} eval users…")
        t_score = time.time()
        score_matrix = model.score_users(eval_user_idxs)  # [n_eval × n_items]
        t_score = time.time() - t_score
        print(f"  score time: {t_score:.1f}s  "
              f"(matrix: {score_matrix.shape}, {score_matrix.nbytes / 1e6:.0f} MB)")

        metrics = compute_metrics(score_matrix, target_item_idxs, user_seen_idxs)
        results[model_name] = metrics
        print(f"  recall@10={metrics['recall@10']:.6f}  "
              f"ndcg@10={metrics['ndcg@10']:.6f}")

        save_scores(
            model_name, fold_name,
            score_matrix, target_item_idxs, user_seen_idxs,
            Config.SCORES_DIR,
        )
        print(f"  → scores cached to artifacts/scores/{model_name}_{fold_name}.npy")

    print()
    print_leaderboard(results, fold=fold_name)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",   default="a", choices=["a", "b"])
    parser.add_argument("--models", nargs="+",   default=None,
                        help="Subset of models to train (default: all)")
    args = parser.parse_args()
    run(fold_name=args.fold, model_names=args.models)


if __name__ == "__main__":
    main()
