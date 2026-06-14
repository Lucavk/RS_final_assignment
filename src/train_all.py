from __future__ import annotations
from src.evaluate import print_leaderboard
from src.metrics import compute_metrics
from src.splits import fold_a, fold_b, val_targets_to_arrays
from src.data import (
    DataBundle,
    build_bundle,
    build_id_maps,
    load_all_data,
    load_train_only,
    load_submission_user_ids,
)
from src.config import Config, RANDOM_SEED
import numpy as np

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


np.random.seed(RANDOM_SEED)


# Score caching helpers.

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
    meta = {"target_item_idxs": target_item_idxs,
            "user_seen_idxs": user_seen_idxs}
    with open(scores_dir / f"{model_name}_{fold}_meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=4)


def load_best_params(model_name: str) -> dict:
    # Load tuned hyperparams if they exist.
    path = Config.PARAMS_DIR / f"{model_name}_best.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"  [params] loaded tuned params for {model_name}: {params}")
        return params
    return {}


# Model factory.

def build_model(model_name: str, params: dict | None = None):
    # Instantiate the model with tuned params where available.
    p = params or load_best_params(model_name)

    if model_name == "popularity":
        from src.models.popularity import PopularityRecommender
        return PopularityRecommender(
            halflife_days=p.get(
                "halflife_days", Config.POPULARITY_HALFLIFE_DAYS)
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
    if model_name == "bpr":
        from src.models.bpr import BPRRecommender
        return BPRRecommender(
            factors=p.get("factors", Config.BPR_FACTORS),
            learning_rate=p.get("learning_rate", Config.BPR_LEARNING_RATE),
            regularization=p.get("regularization", Config.BPR_REGULARIZATION),
            iterations=p.get("iterations", Config.BPR_ITERATIONS),
            random_state=RANDOM_SEED,
        )
    if model_name == "multvae":
        from src.models.multvae import MultVAERecommender
        return MultVAERecommender(
            hidden=p.get("hidden", Config.VAE_HIDDEN),
            latent=p.get("latent", Config.VAE_LATENT),
            dropout=p.get("dropout", Config.VAE_DROPOUT),
            lr=p.get("lr", Config.VAE_LR),
            weight_decay=p.get("weight_decay", Config.VAE_WEIGHT_DECAY),
            batch_size=Config.VAE_BATCH_SIZE,
            epochs=p.get("epochs", Config.VAE_EPOCHS),
            beta=p.get("beta", Config.VAE_BETA),
            anneal_epochs=Config.VAE_ANNEAL_EPOCHS,
            random_state=RANDOM_SEED,
        )
    if model_name == "content":
        from src.models.content_knn import ContentKNNRecommender
        return ContentKNNRecommender(
            topk=p.get("topk", Config.CONTENT_TOPK),
            max_features=Config.CONTENT_MAX_FEATURES,
            min_df=Config.CONTENT_MIN_DF,
        )
    if model_name == "lightgcn":
        from src.models.lightgcn import LightGCNRecommender
        return LightGCNRecommender(
            embedding_dim=p.get("embedding_dim", Config.LIGHTGCN_DIM),
            num_layers=p.get("num_layers", Config.LIGHTGCN_LAYERS),
            epochs=p.get("epochs", Config.LIGHTGCN_EPOCHS),
            batch_size=Config.LIGHTGCN_BATCH_SIZE,
            lr=p.get("lr", Config.LIGHTGCN_LR),
            weight_decay=p.get("weight_decay", Config.LIGHTGCN_WEIGHT_DECAY),
            random_state=RANDOM_SEED,
        )
    if model_name == "bert4rec":
        from src.models.bert4rec import Bert4RecRecommender
        return Bert4RecRecommender(
            max_seq_len=p.get("max_seq_len", Config.BERT4REC_MAX_SEQ_LEN),
            embedding_dim=p.get("embedding_dim", Config.BERT4REC_DIM),
            num_heads=Config.BERT4REC_HEADS,
            num_layers=p.get("num_layers", Config.BERT4REC_LAYERS),
            dropout=p.get("dropout", Config.BERT4REC_DROPOUT),
            epochs=p.get("epochs", Config.BERT4REC_EPOCHS),
            batch_size=Config.BERT4REC_BATCH_SIZE,
            lr=p.get("lr", Config.BERT4REC_LR),
            random_state=RANDOM_SEED,
        )
    if model_name == "recency":
        from src.models.recency import RecencyTransitionRecommender
        return RecencyTransitionRecommender(
            window=p.get("window", Config.RECENCY_WINDOW),
            decay=p.get("decay", Config.RECENCY_DECAY),
            recency_decay=p.get("recency_decay", Config.RECENCY_RECENCY_DECAY),
            shrinkage=p.get("shrinkage", Config.RECENCY_SHRINKAGE),
            pop_discount=p.get("pop_discount", Config.RECENCY_POP_DISCOUNT),
            max_recent=Config.RECENCY_MAX_RECENT,
        )
    raise ValueError(f"Unknown model: {model_name}")


# Main training loop.

ALL_MODELS = ["popularity", "itemknn", "ease", "als", "bpr", "multvae",
              "content", "lightgcn", "bert4rec", "recency"]


def run(fold_name: str = "a", model_names: list | None = None, df_full=None,
        train_only: bool = True) -> dict:
    # Train each model on a fold and cache its score matrix.
    if model_names is None:
        model_names = ALL_MODELS

    if df_full is None:
        if train_only:
            print("Loading train.csv only (honest local validation)...")
            df_full = load_train_only(Config)
        else:
            print("Loading combined train+test data...")
            df_full = load_all_data(Config)

    sub_ids = load_submission_user_ids(Config)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(df_full)

    print(f"\nBuilding Fold {fold_name.upper()}...")
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

    if train_only:
        print("  (validation trained on train.csv only; honest estimate, no time-leak)")

    # Align targets with matrix indices.
    eval_user_idxs, target_item_idxs = val_targets_to_arrays(
        val_targets, user_to_idx, item_to_idx
    )

    # Use train-only history for evaluation masking.
    user_seen_idxs = [
        train_bundle.user_seen_idxs.get(u_idx, set())
        for u_idx in eval_user_idxs
    ]

    results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training  {model_name.upper()}  (fold={fold_name})...")
        print(f"{'='*60}")

        t_start = time.time()
        model = build_model(model_name)

        # Popularity can use time decay when timestamps are available.
        if model_name == "popularity" and model.halflife_days is not None:
            model.fit_with_decay(train_bundle, train_df)
        else:
            model.fit(train_bundle)

        t_fit = time.time() - t_start
        print(f"  fit time: {t_fit:.1f}s")

        print(f"  scoring {len(eval_user_idxs):,} eval users...")
        t_score = time.time()
        score_matrix = model.score_users(eval_user_idxs)
        t_score = time.time() - t_score
        print(f"  score time: {t_score:.1f}s  "
              f"(matrix: {score_matrix.shape}, {score_matrix.nbytes / 1e6:.0f} MB)")

        metrics = compute_metrics(
            score_matrix, target_item_idxs, user_seen_idxs)
        results[model_name] = metrics
        print(f"  recall@10={metrics['recall@10']:.6f}  "
              f"ndcg@10={metrics['ndcg@10']:.6f}")

        save_scores(
            model_name, fold_name,
            score_matrix, target_item_idxs, user_seen_idxs,
            Config.SCORES_DIR,
        )
        print(
            f"  -> scores cached to artifacts/scores/{model_name}_{fold_name}.npy")

    print()
    print_leaderboard(results, fold=fold_name)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       default="a", choices=["a", "b"])
    parser.add_argument("--models",     nargs="+",   default=None,
                        help="Subset of models to train (default: all)")
    parser.add_argument("--combined",   action="store_true",
                        help="Use train+test combined (inflates local score; don't use)")
    args = parser.parse_args()
    run(fold_name=args.fold, model_names=args.models, train_only=not args.combined)


if __name__ == "__main__":
    main()
