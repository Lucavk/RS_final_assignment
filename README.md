# Recommender Systems — Final Assignment

This repository contains our final recommender system pipeline for the Kaggle assignment.
The evaluation metric is **Recall@10** on the hidden Kaggle test set.

The final model is an ensemble of several collaborative filtering approaches:

* EASE
* ALS
* ItemKNN
* Popularity baseline

There are also extra models in the codebase, such as BPR, MultVAE, and ContentKNN, which can be tuned and evaluated as well.



## 1. Setup

The base environment on the university system does not include all required packages, so it is easiest to create a separate conda environment.

```bash
conda create -n recsys python=3.11 -y
conda activate recsys
pip install -r requirements.txt
```

If you want to use CUDA on the university GPU node, install the CUDA PyTorch wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

On an M1/M2 Mac, the normal PyPI PyTorch wheel should already support MPS.



## 2. Data

Put the Kaggle files in the `data/` folder:

```text
data/
  train.csv
  test.csv
  item_meta.csv
  sample_submission.csv
```

Important: both `train.csv` and `test.csv` are input history.
The Kaggle labels are not inside these files. Kaggle removed the final interaction for each of the 2,255 submission users, and those hidden interactions are what the leaderboard evaluates.

So for the final submission, the model is trained using both:

* `train.csv`, which contains interactions up to 2021-04-20
* `test.csv`, which contains later interaction history

There is no ID-level cold start issue: all submission users and items already appear in the provided data.



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

Runtime is roughly:

* around 2–3 hours on an M1 Max
* around 1 hour on the university GPU node



## 4. Step-by-step pipeline

Running the pipeline step by step is usually better, because it makes it easier to inspect results and debug problems.

### Step 1 — Tune the models

We tune on Fold B, because it uses many more users and gives a more stable validation signal.

```bash
python src/tune.py --model ease       --fold b
python src/tune.py --model itemknn    --fold b
python src/tune.py --model als        --fold b
python src/tune.py --model bpr        --fold b
python src/tune.py --model multvae    --fold b
python src/tune.py --model content    --fold b
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

Re-running tuning continues the existing Optuna study instead of starting from scratch.

The default number of trials is set in `Config.TUNE_N_TRIALS`, but it can be overridden manually:

```bash
python src/tune.py --model ease --fold b --n_trials 100
```



### Step 2 — Train all models and cache scores

After tuning, train all models and cache their score matrices.

```bash
python src/train_all.py --fold b
python src/train_all.py --fold a
```

Fold B is mainly used for ensemble tuning.
Fold A is the more realistic check, because it mirrors the Kaggle setup more closely.

The cached scores are written to:

```text
artifacts/scores/
```

Each run also prints a small leaderboard with Recall@10 and NDCG@10.



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

The script also validates the file format automatically.
The expected output is 2,255 rows with 10 recommendations per user.



## 5. Quick evaluation

To quickly train and evaluate the default models on Fold A:

```bash
python src/train_all.py --fold a
```

To evaluate specific cached models:

```bash
python src/evaluate.py --fold a --models ease als itemknn popularity
```



## 6. Validation setup

The project uses two leave-one-out validation folds.

### Fold A — submission-style split

Fold A holds out the last interaction of each of the 2,255 submission users.

This is the most important validation fold, because it is designed to match the Kaggle task as closely as possible. It should be used for the final sanity check before submitting.

### Fold B — global split

Fold B holds out the last interaction of every user, around 22k users in total.

This fold has lower variance, so it is more useful for hyperparameter tuning and ensemble weight search. For speed, tuning uses a 5k-user subsample of Fold B.



## 7. Models

Several recommender models were included in the pipeline, each capturing a slightly different pattern in the interaction data. This was useful because a single model usually focuses on one type of signal, while an ensemble can combine multiple perspectives.

EASE was one of the main collaborative filtering models. It works by learning relationships between items and can recommend products based on which items tend to be predictive of each other. This makes it a strong baseline for item-to-item recommendation tasks.

ItemKNN follows a similar idea, but in a more direct neighbourhood-based way. It recommends items that often occur together with items the user has interacted with before. Although it is a relatively simple method, it is useful because it captures clear co-occurrence patterns in the data.

ALS was included to capture broader user and item preference patterns. Instead of only looking at direct item similarities, it factorises the interaction matrix into user and item representations. This allows the model to recommend items based on more general preference structures.

BPR is also a matrix factorisation model, but it is trained as a ranking model. Its goal is to rank items that a user interacted with above items they did not interact with. This makes it suitable for a top-10 recommendation task, where the ordering of recommended items is important.

MultVAE adds a neural model to the pipeline. It tries to reconstruct a user’s interaction profile and then uses that reconstruction to predict which unseen items are likely to be relevant. This gives the ensemble a different type of signal compared with the more traditional collaborative filtering models.

ContentKNN uses the metadata from `item_meta.csv`. Instead of only relying on user-item interactions, it looks at item features and recommends items that are similar in terms of their content. No external pretrained embeddings were used, so the model stays within the competition rules.

Finally, a popularity model was included as a simple fallback. It recommends items that are popular, with more weight given to recent interactions. Even though this model is basic, it can help stabilise the ensemble and perform well for users where the more complex models are less confident.

In the final ensemble, the strongest models were combined because they bring different types of information: item similarity, matrix factorisation, content-based similarity, and recent popularity. This made the recommendations more robust than relying on one model family only.


## 8. Project structure

```text
src/
  config.py          paths, seeds, default hyperparameters
  data.py            data loading, cleaning, ID mapping, CSR matrix creation
  splits.py          Fold A and Fold B validation splits
  metrics.py         Recall@k and NDCG@k
  models/
    base.py          shared recommender interface
    popularity.py
    ease.py
    itemknn.py
    als.py
  ensemble/
    blend.py         ensemble weight optimisation
  evaluate.py        evaluate cached scores
  train_all.py       train models and cache score matrices
  tune.py            Optuna tuning script
  submit.py          final submission script

artifacts/
  scores/            cached score matrices
  params/            tuned parameters and ensemble weights
  optuna/            Optuna SQLite studies

data/
  submission.csv     generated Kaggle submission
```



## 9. Reproducibility

The random seed is set to `42` across NumPy, PyTorch, implicit, and Optuna where applicable.

Most configuration is kept in:

```text
src/config.py
```

Given the same data, seed, and environment, the pipeline should produce the same artifacts. Optuna studies are saved in `artifacts/optuna/`, so previous tuning runs can be inspected or continued later.
