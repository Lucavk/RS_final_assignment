# Recommender Systems — Final Assignment

This repository contains our final recommender system pipeline for the Kaggle assignment.
The evaluation metric is **Recall@10** on the hidden Kaggle test set.

## 1. Setup

The base environment on the university system does not include all required packages, so it is easiest to create a separate conda environment.

```bash
conda create -n recsys python=3.11 -y
conda activate recsys
pip install -r requirements.txt
```

If you want to use CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```



## 2. Data

Put the Kaggle files in the `data/` folder:

```text
data/
  train.csv
  test.csv
  item_meta.csv
  sample_submission.csv
```

## 3. Full pipeline

To run the full pipeline in one go:

```bash
conda activate recsys
bash run.sh
```

This runs:

1. hyperparameter tuning
2. model training
3. ensemble weight tuning
4. submission generation

If something does not work you can use:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export RS_SCORES_DIR="path_to_local_storage_instead_of_/home"
```

## 4. Step-by-step pipeline

### Step 1 — Tune the models

We tune on Fold B, because it uses many more users and gives a more stable validation signal.

```bash
python src/tune.py --model ease       --fold b
python src/tune.py --model itemknn    --fold b
python src/tune.py --model als        --fold b
python src/tune.py --model bpr        --fold b
python src/tune.py --model multvae    --fold b
python src/tune.py --model content    --fold b
python src/tune.py --model lightgcn   --fold b
python src/tune.py --model bert4rec   --fold b
python src/tune.py --model recency    --fold b
python src/tune.py --model popularity --fold b
```

The best parameters are saved in:

```text
artifacts/params/<model>_best.json
```

Optuna studies are stored in:

```text
artifacts/optuna/<model>.db
```

### Step 2 — Train all models and cache scores

After tuning, train all models and cache their score matrices.

```bash
python src/train_all.py --fold b
python src/train_all.py --fold a
```

The cached scores are written to:

```text
artifacts/scores/
```


### Step 3 — Tune ensemble weights

The ensemble weights are tuned on Fold B and checked on Fold A.

```bash
python src/ensemble/blend.py --tune_fold b --check_fold a --n_trials 300
```

The resulting weights are saved to:

```text
artifacts/params/ensemble_weights.json
```



### Step 4 — Create the final submission

Generate the Kaggle submission file with:

```bash
python src/submit.py
```

This writes:

```text
data/submission.csv
```
