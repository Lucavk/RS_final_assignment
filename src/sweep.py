from __future__ import annotations
from src.train_all import build_model
from src.submit import rrf_blend, top_k_with_fallback
from src.ensemble.blend import precompute_rrf, fast_recall, blend_weighted
from src.evaluate import load_scores
from src.data import build_bundle, build_id_maps, load_all_data, load_submission_user_ids
from src.config import Config, RANDOM_SEED
import pandas as pd
import numpy as np

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


np.random.seed(RANDOM_SEED)


# Tune weights for one candidate

def tune_weights(rrf_tune, targets, seen, model_names, n_trials):
    # Search non-negative model weights
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        w = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in model_names]
        if sum(w) < 1e-9:
            return 0.0
        return fast_recall(blend_weighted(rrf_tune, w), targets, seen)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.enqueue_trial({f"w_{m}": 1.0 for m in model_names})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return [study.best_params[f"w_{m}"] for m in model_names]


# Main sweep.

def run(model_names, rrf_ks, n_trials, max_eval_users, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cached fold scores for tuning and validation
    print("Loading cached Fold B / Fold A scores...")
    b_scores, tgt_b, seen_b = {}, None, None
    a_scores, tgt_a, seen_a = {}, None, None
    for m in model_names:
        s, t, se = load_scores(m, "b")
        b_scores[m] = s
        if tgt_b is None:
            tgt_b, seen_b = t, se
        s, t, se = load_scores(m, "a")
        a_scores[m] = s
        if tgt_a is None:
            tgt_a, seen_a = t, se

    # Subsample Fold B for fast weight tuning.
    n_b = b_scores[model_names[0]].shape[0]
    if n_b > max_eval_users:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = rng.choice(n_b, max_eval_users, replace=False)
        b_scores = {m: s[sel] for m, s in b_scores.items()}
        tgt_b = [tgt_b[i] for i in sel]
        seen_b = [seen_b[i] for i in sel]

    # Train each model once on full data
    print("\nTraining each model once on full data (train+test)...")
    df_full = load_all_data(Config)
    sub_ids = load_submission_user_ids(Config)
    u2i, i2u, it2i, i2it = build_id_maps(df_full)
    full_bundle = build_bundle(df_full, u2i, i2u, it2i, i2it, sub_ids)
    sub_idxs = np.array([u2i[u] for u in sub_ids if u in u2i], dtype=np.int32)
    seen_sub = [full_bundle.user_seen_idxs.get(
        int(u), set()) for u in sub_idxs]

    sub_scores, pop_scores = {}, None
    for m in model_names:
        t0 = time.time()
        model = build_model(m)
        if m == "popularity" and getattr(model, "halflife_days", None) is not None:
            model.fit_with_decay(full_bundle, df_full)
        else:
            model.fit(full_bundle)
        sub_scores[m] = model.score_users(sub_idxs)
        if m == "popularity":
            pop_scores = model._item_scores
        print(f"  {m:<10} {time.time()-t0:5.1f}s")
    if pop_scores is None:
        pop_scores = np.asarray(full_bundle.train_matrix.sum(axis=0)).ravel()

    # Define candidate ensembles
    candidates = []
    for k in rrf_ks:
        candidates.append((f"all_k{k}", model_names, k, "tuned"))
    candidates.append(("all_k60_equal", model_names, 60, "equal"))
    for m in model_names:
        subset = [x for x in model_names if x != m]
        candidates.append((f"drop_{m}_k60", subset, 60, "tuned"))

    # Evaluate and build each candidate
    print(f"\nEvaluating {len(candidates)} candidates...\n")
    results = []
    for name, subset, k, mode in candidates:
        # Choose weights for this candidate
        if mode == "equal":
            weights = [1.0] * len(subset)
        else:
            rrf_b = precompute_rrf([b_scores[m] for m in subset], k)
            weights = tune_weights(rrf_b, tgt_b, seen_b, subset, n_trials)

        # Score the candidate on Fold A
        rrf_a = precompute_rrf([a_scores[m] for m in subset], k)
        recall_a = fast_recall(blend_weighted(rrf_a, weights), tgt_a, seen_a)

        # Write the submission file
        blended = rrf_blend([sub_scores[m] for m in subset], weights, rrf_k=k)
        recs = top_k_with_fallback(
            blended, seen_sub, pop_scores, i2it, k=Config.K)
        _write_submission(recs, sub_ids, u2i, out_dir / f"{name}.csv")

        results.append((name, recall_a))
        print(f"  {name:<22} foldA recall@10 = {recall_a:.6f}")

    # Save the ranked summary
    results.sort(key=lambda x: x[1], reverse=True)
    rank_df = pd.DataFrame(results, columns=["candidate", "foldA_recall@10"])
    rank_df.to_csv(out_dir / "_ranking.csv", index=False)

    print("\n" + "=" * 50)
    print("Ranked candidates (by local Fold-A recall@10)")
    print("=" * 50)
    for i, (name, r) in enumerate(results, 1):
        print(f"{i:2d}. {name:<22} {r:.6f}")
    print("=" * 50)
    print(
        f"\nSubmissions in {out_dir}/  -  submit the top few by Fold-A score.")


def _write_submission(recs_list, sub_ids, user_to_idx, path):
    # Write recommendations to the Kaggle CSV format.
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    found = [u for u in sub_ids if u in user_to_idx]
    id_to_rec = {uid: recs for uid, recs in zip(found, recs_list)}
    sub_df["item_id"] = sub_df["user_id"].apply(
        lambda u: ",".join(str(r) for r in id_to_rec.get(u, [])[:Config.K])
    )
    sub_df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=Config.ENSEMBLE_MODELS)
    parser.add_argument("--rrf_ks", nargs="+", type=int,
                        default=[20, 40, 60, 100])
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--max_eval_users", type=int, default=10000)
    parser.add_argument("--out_dir", default=str(Config.DATA_DIR / "sweep"))
    args = parser.parse_args()
    run(args.models, args.rrf_ks, args.n_trials,
        args.max_eval_users, Path(args.out_dir))


if __name__ == "__main__":
    main()
