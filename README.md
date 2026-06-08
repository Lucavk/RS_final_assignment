# Recommender Systems — Final Assignment

Kaggle metric: **Recall@10** on a hidden test set.  
Architecture: ensemble of Tier-1 CF models (EASE, ALS, ItemKNN, Popularity).

---

## Environment setup

Create a dedicated conda environment (the base env lacks pandas/implicit):

```bash
conda create -n recsys python=3.11 -y
conda activate recsys
pip install -r requirements.txt
```

> **PyTorch note:** if you need GPU (CUDA) support on the university node:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> For M1 Mac the wheel from PyPI already includes MPS support.

---

## Data

Place the three Kaggle files in `data/`:

```
data/
  train.csv
  test.csv
  item_meta.csv
  sample_submission.csv
```

Key data facts:
- `train.csv` (2002–2021-04-20) and `test.csv` (2021-04-21 onward) are both **input** history — both are used for training.
- The Kaggle ground truth = the chronologically **last interaction** of each of the 2,255 submission users (removed by Kaggle from the provided data).
- All submission users and all items are present in the provided data (no cold-start at ID level).

---

## Running the pipeline

### Option A — full automated pipeline (one night on the GPU node)

```bash
conda activate recsys
bash run.sh
```

This runs all four stages: tune → train → blend → submit.  
Expected runtime: ~2–3 h on M1 Max, ~1 h on the GPU node.

### Option B — step by step (recommended for control)

**Step 1: Tune each model (Fold B, low-variance)**

```bash
python src/tune.py --model ease       --fold b --n_trials 15
python src/tune.py --model itemknn    --fold b --n_trials 40
python src/tune.py --model als        --fold b --n_trials 50
python src/tune.py --model popularity --fold b --n_trials 10
```

Best params are saved to `artifacts/params/<model>_best.json`.  
Optuna studies persist in `artifacts/optuna/<model>.db` — re-running adds more trials.

**Step 2: Train on Fold A, cache score matrices**

```bash
python src/train_all.py --fold a
```

Prints a leaderboard table (Recall@10, NDCG@10) for all models.  
Score matrices are cached in `artifacts/scores/`.

**Step 3: Optimise ensemble blend weights**

```bash
python src/ensemble/blend.py --fold a --n_trials 200
```

Saves tuned weights to `artifacts/params/ensemble_weights.json`.

**Step 4: Generate submission**

```bash
python src/submit.py
```

Writes `data/submission.csv`. Validated automatically (2,255 rows × 10 items).

---

## Individual scripts

| Script | Purpose |
|--------|---------|
| `src/tune.py` | Optuna tuning for a single model |
| `src/train_all.py` | Train all models on a fold, cache scores |
| `src/evaluate.py` | Print leaderboard from cached scores |
| `src/ensemble/blend.py` | Optimise ensemble weights |
| `src/submit.py` | Generate final submission CSV |

---

## Quick eval (without tuning)

Train with default hyperparameters and see Fold A results immediately:

```bash
python src/train_all.py --fold a
```

To evaluate individual models from cached scores:

```bash
python src/evaluate.py --fold a --models ease als itemknn popularity
```

---

## Validation design

- **Fold A** ("submission LOO"): hold out the last interaction of each of the 2,255 submission users. This **exactly mirrors the Kaggle task**. Use this for final model selection.
- **Fold B** ("global LOO"): hold out the last interaction of every user (~22k). Lower variance → better for hyperparameter search. Tuning uses a 5k-user subsample of Fold B for speed.

---

## Model descriptions

| Model | File | Key hyperparameters |
|-------|------|---------------------|
| EASE^R | `src/models/ease.py` | `lam` (L2 reg, 50–2000) |
| Item-item KNN | `src/models/itemknn.py` | `topk` (50–500), `shrinkage` (0–500) |
| ALS | `src/models/als.py` | `factors`, `regularization`, `alpha`, `iterations` |
| Popularity | `src/models/popularity.py` | `halflife_days` (recency decay) |

All content features are derived **only from the provided data** (no external pretrained embeddings).

---

## File structure

```
src/
  config.py          all paths, seeds, default hyperparams
  data.py            DataBundle: load, clean, ID maps, CSR matrix, sequences
  splits.py          Fold A (submission LOO) + Fold B (global LOO)
  metrics.py         vectorised Recall@k, NDCG@k
  models/
    base.py          Recommender interface (fit / score_users / recommend)
    popularity.py
    ease.py
    itemknn.py
    als.py
  ensemble/
    blend.py         RRF weight optimiser
  evaluate.py        leaderboard table
  train_all.py       train models, cache score matrices
  tune.py            Optuna driver
  submit.py          final submission generator
artifacts/           (gitignored, created automatically)
  scores/            cached score matrices (.npy)
  params/            best hyperparams (.json)
  optuna/            Optuna SQLite databases (.db)
data/
  submission.csv     output (upload to Kaggle)
```

---

## Reproducibility

- Random seed: `42` everywhere (numpy, torch, implicit, Optuna).
- All model hyperparameters are in `src/config.py`.
- Artifacts are deterministic given the same seed and data.
- Optuna studies in `artifacts/optuna/` can be inspected for tuning analysis.
