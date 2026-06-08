"""
Generate the final Kaggle submission.

Steps:
1. Retrain each ensemble model on the FULL dataset (train + test combined).
2. Score the 2,255 submission users.
3. Blend scores via Reciprocal Rank Fusion (RRF) using tuned weights.
4. Filter already-seen items, take top-10, backfill with popularity.
5. Write data/submission.csv  in the required format:
       ID,user_id,item_id
       12,12,"6312,12419,6891,664,4243,8377,7962,6635,12842,4970"
       ...

Usage:
    python src/submit.py                         # RRF ensemble of all models
    python src/submit.py --models ease als       # only these two models
    python src/submit.py --model ease            # single best model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, RANDOM_SEED
from src.data import build_bundle, build_id_maps, load_all_data, load_submission_user_ids
from src.train_all import build_model

np.random.seed(RANDOM_SEED)


# ── RRF blending ─────────────────────────────────────────────────────────────

def rrf_blend(score_matrices: list[np.ndarray], weights: list[float], rrf_k: int = 60) -> np.ndarray:
    """
    Reciprocal Rank Fusion.
    score_matrices: list of [U × n_items] float32 arrays
    weights: per-model weight (will be normalised to sum to 1)
    Returns blended score matrix [U × n_items].
    """
    total_weights = sum(weights)
    blended = np.zeros_like(score_matrices[0])

    for mat, w in zip(score_matrices, weights):
        # Rank each item per user (1 = best)
        # argsort twice to get ranks
        order = np.argsort(-mat, axis=1)          # [U × n_items], item idx sorted by score
        ranks = np.empty_like(order)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)
        rrf_scores = 1.0 / (rrf_k + ranks)
        blended += (w / total_weights) * rrf_scores

    return blended


# ── Top-k extraction with fallback ───────────────────────────────────────────

def top_k_with_fallback(
    blended_scores: np.ndarray,
    user_seen_idxs: list,
    popularity_scores: np.ndarray,
    idx_to_item: dict,
    k: int = 10,
) -> list[list]:
    """
    For each user row:
      1. Mask seen items (set to -inf).
      2. Take top-k by blended score.
      3. If fewer than k remain, backfill with unseen popular items.
    Returns a list of lists of original item_id strings.
    """
    n_users, n_items = blended_scores.shape
    recommendations = []

    # Precompute global popularity order for fallback
    pop_order = np.argsort(-popularity_scores)  # [n_items]

    for i in range(n_users):
        scores = blended_scores[i].copy()
        seen   = user_seen_idxs[i]

        if seen:
            scores[list(seen)] = -np.inf

        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        recs = [idx for idx in top_idxs if scores[idx] != -np.inf]

        # Backfill with popularity if needed
        if len(recs) < k:
            existing = set(recs) | seen
            for pop_idx in pop_order:
                if pop_idx not in existing:
                    recs.append(int(pop_idx))
                    existing.add(int(pop_idx))
                if len(recs) == k:
                    break

        recommendations.append([idx_to_item[idx] for idx in recs[:k]])

    return recommendations


# ── Main submission pipeline ──────────────────────────────────────────────────

def run(model_names: list | None = None):
    if model_names is None:
        model_names = Config.ENSEMBLE_MODELS

    print("=" * 60)
    print("SUBMISSION GENERATION")
    print("=" * 60)

    # ── Full dataset ──────────────────────────────────────────────────────────
    print("\nLoading full dataset (train + test)…")
    df_full = load_all_data(Config)
    sub_ids = load_submission_user_ids(Config)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(df_full)

    full_bundle = build_bundle(
        df_full, user_to_idx, idx_to_user, item_to_idx, idx_to_item, sub_ids
    )

    print(f"  users={full_bundle.n_users:,}  items={full_bundle.n_items:,}")
    print(f"  submission users={len(sub_ids):,}")

    # Submission user indices — these are the rows we predict for
    sub_idxs = np.array(
        [user_to_idx[u] for u in sub_ids if u in user_to_idx],
        dtype=np.int32,
    )
    print(f"  submission users found in data: {len(sub_idxs):,}/{len(sub_ids):,}")

    # Seen-item sets per submission user (full history — filter from recs)
    user_seen_idxs = [
        full_bundle.user_seen_idxs.get(u_idx, set())
        for u_idx in sub_idxs
    ]

    # ── Train each model on full data ─────────────────────────────────────────
    score_matrices = []
    model_weights  = []
    pop_scores     = None

    for model_name in model_names:
        print(f"\n{'─'*40}")
        print(f"Training  {model_name.upper()}  on full data…")
        t0 = time.time()

        model = build_model(model_name)

        if model_name == "popularity" and model.halflife_days is not None:
            model.fit_with_decay(full_bundle, df_full)
        else:
            model.fit(full_bundle)

        print(f"  fit time: {time.time()-t0:.1f}s")

        t0 = time.time()
        scores = model.score_users(sub_idxs)  # [n_sub × n_items]
        print(f"  score time: {time.time()-t0:.1f}s")

        score_matrices.append(scores)
        model_weights.append(1.0)   # equal weights; override with tuned weights if available

        # Keep popularity scores for fallback
        if model_name == "popularity":
            pop_scores = model._item_scores

    # Load tuned ensemble weights if available (put in artifacts/params/ensemble_weights.json)
    weights_path = Config.PARAMS_DIR / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights_dict = json.load(f)
        model_weights = [weights_dict.get(m, 1.0) for m in model_names]
        print(f"\nLoaded ensemble weights: {dict(zip(model_names, model_weights))}")
    else:
        print(f"\nUsing equal weights (no {weights_path} found)")

    if pop_scores is None:
        # Fallback: uniform popularity
        pop_scores = np.asarray(full_bundle.train_matrix.sum(axis=0)).ravel()

    # ── Blend ─────────────────────────────────────────────────────────────────
    print("\nBlending scores via RRF…")
    blended = rrf_blend(score_matrices, model_weights, rrf_k=Config.ENSEMBLE_RRF_K)

    # ── Top-10 per user ───────────────────────────────────────────────────────
    print("Extracting top-10 recommendations…")
    recs_list = top_k_with_fallback(
        blended, user_seen_idxs, pop_scores, idx_to_item, k=Config.K
    )

    # ── Format submission CSV ─────────────────────────────────────────────────
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Build a mapping user_id -> recommendations
    sub_id_to_rec = {}
    for uid, recs in zip([u for u in sub_ids if u in user_to_idx], recs_list):
        sub_id_to_rec[uid] = recs

    # Fill in recommendations in sample_submission order (preserves original IDs)
    def _format_recs(uid):
        recs = sub_id_to_rec.get(uid, [])
        if not recs:
            # Cold user not found — use global popularity
            pop_order = np.argsort(-pop_scores)
            seen = full_bundle.user_seen_idxs.get(user_to_idx.get(uid, -1), set())
            recs = [idx_to_item[i] for i in pop_order if i not in seen][:Config.K]
        return ",".join(str(r) for r in recs[:Config.K])

    sub_df["item_id"] = sub_df["user_id"].apply(_format_recs)

    out_path = Config.SUBMISSION_OUT_PATH
    sub_df.to_csv(out_path, index=False)
    print(f"\nSubmission written to  {out_path}")

    # ── Validation checks ─────────────────────────────────────────────────────
    print("\nValidation:")
    assert len(sub_df) == len(pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)), \
        "Row count mismatch!"
    for _, row in sub_df.iterrows():
        items = str(row["item_id"]).split(",")
        assert len(items) == Config.K, \
            f"User {row['user_id']} has {len(items)} recs (expected {Config.K})"
        assert len(set(items)) == Config.K, \
            f"Duplicate items for user {row['user_id']}"
    print(f"  ✓ {len(sub_df):,} rows, all with exactly {Config.K} unique items")
    print("Ready to submit!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to ensemble (default: Config.ENSEMBLE_MODELS)")
    args = parser.parse_args()
    run(model_names=args.models)


if __name__ == "__main__":
    main()
