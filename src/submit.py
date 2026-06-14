from __future__ import annotations
from src.train_all import build_model
from src.data import build_bundle, build_id_maps, load_all_data, load_submission_user_ids
from src.config import Config, RANDOM_SEED
import pandas as pd
import numpy as np

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


np.random.seed(RANDOM_SEED)


# RRF blending.

def rrf_blend(score_matrices: list[np.ndarray], weights: list[float], rrf_k: int = 60) -> np.ndarray:
    # Blend ranked model outputs into one score matrix
    total_weights = sum(weights)
    blended = np.zeros_like(score_matrices[0])

    for mat, w in zip(score_matrices, weights):
        # Rank each item per user.
        order = np.argsort(-mat, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, mat.shape[1] + 1)
        rrf_scores = 1.0 / (rrf_k + ranks)
        blended += (w / total_weights) * rrf_scores

    return blended


# Top-k extraction with fallback

def top_k_with_fallback(
    blended_scores: np.ndarray,
    user_seen_idxs: list,
    popularity_scores: np.ndarray,
    idx_to_item: dict,
    k: int = 10,
) -> list[list]:
    # Convert blended scores into original item ids
    n_users, n_items = blended_scores.shape
    recommendations = []

    # Precompute global popularity order for fallback
    pop_order = np.argsort(-popularity_scores)

    for i in range(n_users):
        scores = blended_scores[i].copy()
        seen = user_seen_idxs[i]

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


# Main submission pipeline

def run(model_names: list | None = None):
    if model_names is None:
        model_names = Config.ENSEMBLE_MODELS

    print("=" * 60)
    print("SUBMISSION GENERATION")
    print("=" * 60)

    print("\nLoading full dataset (train + test)...")
    df_full = load_all_data(Config)
    sub_ids = load_submission_user_ids(Config)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(df_full)

    full_bundle = build_bundle(
        df_full, user_to_idx, idx_to_user, item_to_idx, idx_to_item, sub_ids
    )

    print(f"  users={full_bundle.n_users:,}  items={full_bundle.n_items:,}")
    print(f"  submission users={len(sub_ids):,}")

    # These are the user rows we predict for
    sub_idxs = np.array(
        [user_to_idx[u] for u in sub_ids if u in user_to_idx],
        dtype=np.int32,
    )
    print(
        f"  submission users found in data: {len(sub_idxs):,}/{len(sub_ids):,}")

    # Filter recommendations against each user's full history
    user_seen_idxs = [
        full_bundle.user_seen_idxs.get(u_idx, set())
        for u_idx in sub_idxs
    ]

    score_matrices = []
    model_weights = []
    pop_scores = None

    for model_name in model_names:
        print(f"\n{'-'*40}")
        print(f"Training  {model_name.upper()}  on full data...")
        t0 = time.time()

        model = build_model(model_name)

        if model_name == "popularity" and model.halflife_days is not None:
            model.fit_with_decay(full_bundle, df_full)
        else:
            model.fit(full_bundle)

        print(f"  fit time: {time.time()-t0:.1f}s")

        t0 = time.time()
        scores = model.score_users(sub_idxs)
        print(f"  score time: {time.time()-t0:.1f}s")

        score_matrices.append(scores)
        model_weights.append(1.0)

        # Keep popularity scores for fallback
        if model_name == "popularity":
            pop_scores = model._item_scores

    # Load tuned ensemble weights if available
    weights_path = Config.PARAMS_DIR / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights_dict = json.load(f)
        model_weights = [weights_dict.get(m, 1.0) for m in model_names]
        print(
            f"\nLoaded ensemble weights: {dict(zip(model_names, model_weights))}")
    else:
        print(f"\nUsing equal weights (no {weights_path} found)")

    if pop_scores is None:
        # Use interaction counts if popularity was not part of the ensemble
        pop_scores = np.asarray(full_bundle.train_matrix.sum(axis=0)).ravel()

    print("\nBlending scores via RRF...")
    blended = rrf_blend(score_matrices, model_weights,
                        rrf_k=Config.ENSEMBLE_RRF_K)

    print("Extracting top-10 recommendations...")
    recs_list = top_k_with_fallback(
        blended, user_seen_idxs, pop_scores, idx_to_item, k=Config.K
    )

    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Map each user id to its recommendations
    sub_id_to_rec = {}
    for uid, recs in zip([u for u in sub_ids if u in user_to_idx], recs_list):
        sub_id_to_rec[uid] = recs

    # Preserve the sample_submission user order
    def _format_recs(uid):
        recs = sub_id_to_rec.get(uid, [])
        if not recs:
            # Use global popularity for cold users
            pop_order = np.argsort(-pop_scores)
            seen = full_bundle.user_seen_idxs.get(
                user_to_idx.get(uid, -1), set())
            recs = [idx_to_item[i]
                    for i in pop_order if i not in seen][:Config.K]
        return ",".join(str(r) for r in recs[:Config.K])

    sub_df["item_id"] = sub_df["user_id"].apply(_format_recs)

    out_path = Config.SUBMISSION_OUT_PATH
    sub_df.to_csv(out_path, index=False)
    print(f"\nSubmission written to  {out_path}")

    print("\nValidation:")
    assert len(sub_df) == len(pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)), \
        "Row count mismatch!"
    for _, row in sub_df.iterrows():
        items = str(row["item_id"]).split(",")
        assert len(items) == Config.K, \
            f"User {row['user_id']} has {len(items)} recs (expected {Config.K})"
        assert len(set(items)) == Config.K, \
            f"Duplicate items for user {row['user_id']}"
    print(
        f"  OK: {len(sub_df):,} rows, all with exactly {Config.K} unique items")
    print("Ready to submit!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to ensemble (default: Config.ENSEMBLE_MODELS)")
    args = parser.parse_args()
    run(model_names=args.models)


if __name__ == "__main__":
    main()
