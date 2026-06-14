from __future__ import annotations
from src.metrics import compute_metrics
from src.config import Config
import numpy as np

import pickle
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_scores(model_name: str, fold: str, scores_dir: Path | None = None):
    # Load a cached score matrix and its evaluation metadata
    if scores_dir is None:
        scores_dir = Config.SCORES_DIR

    scores_path = scores_dir / f"{model_name}_{fold}.npy"
    meta_path = scores_dir / f"{model_name}_{fold}_meta.pkl"

    if not scores_path.exists():
        raise FileNotFoundError(f"Score matrix not found: {scores_path}\n"
                                "Run  python src/train_all.py  first.")

    scores = np.load(scores_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return scores, meta["target_item_idxs"], meta["user_seen_idxs"]


def evaluate_model(model_name: str, fold: str, k: int = 10) -> Dict[str, float]:
    # Load cached scores and compute metrics
    scores, targets, seen = load_scores(model_name, fold)
    return compute_metrics(scores, targets, seen, k=k)


def print_leaderboard(results: Dict[str, Dict[str, float]], fold: str, k: int = 10):
    # Print a sorted results table
    col = f"recall@{k}"
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get(col, 0), reverse=True)

    header = f"{'Model':<14}  {'Recall@'+str(k):>10}  {'NDCG@'+str(k):>10}"
    print()
    print(f"Leaderboard - Fold {fold.upper()}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics in sorted_results:
        print(f"{name:<14}  {metrics.get(col, 0):>10.6f}  "
              f"{metrics.get('ndcg@'+str(k), 0):>10.6f}")
    print("=" * len(header))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="a", choices=["a", "b"])
    parser.add_argument("--models", nargs="+",
                        default=["popularity", "itemknn", "ease", "als"])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    results = {}
    for model_name in args.models:
        try:
            t0 = time.time()
            metrics = evaluate_model(model_name, args.fold, k=args.k)
            elapsed = time.time() - t0
            results[model_name] = metrics
            print(f"{model_name}: recall@{args.k}={metrics[f'recall@{args.k}']:.6f}  "
                  f"ndcg@{args.k}={metrics[f'ndcg@{args.k}']:.6f}  [{elapsed:.1f}s]")
        except FileNotFoundError as e:
            print(f"[SKIP] {model_name}: {e}")

    if results:
        print_leaderboard(results, fold=args.fold, k=args.k)


if __name__ == "__main__":
    main()
