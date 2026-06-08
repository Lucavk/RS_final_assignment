from pathlib import Path

RANDOM_SEED = 42


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    ITEM_META_PATH = DATA_DIR / "item_meta.csv"
    SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_OUT_PATH = DATA_DIR / "submission.csv"

    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    SCORES_DIR = ARTIFACTS_DIR / "scores"
    PARAMS_DIR = ARTIFACTS_DIR / "params"
    OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
    MODELS_DIR = ARTIFACTS_DIR / "models"

    K = 10

    # ── Popularity ──────────────────────────────────────────────────
    # recency_halflife: half-life in days for exponential time-decay (None = no decay)
    POPULARITY_HALFLIFE_DAYS = 365

    # ── Item-item cosine KNN ─────────────────────────────────────────
    ITEMKNN_TOPK = 200         # max neighbors kept per item
    ITEMKNN_SHRINKAGE = 100    # denominator shrinkage (penalises rare co-occurrence)

    # ── EASE^R ───────────────────────────────────────────────────────
    EASE_LAMBDA = 500.0        # L2 regularisation; higher = more regularised

    # ── ALS (implicit) ───────────────────────────────────────────────
    ALS_FACTORS = 128
    ALS_REGULARIZATION = 0.05
    ALS_ITERATIONS = 50
    ALS_ALPHA = 20.0           # confidence scaling: C = 1 + alpha * R

    # ── Tuning ───────────────────────────────────────────────────────
    # Optuna time limits per model (seconds)
    TUNE_TIMEOUT = {
        "popularity": 120,
        "itemknn":    1200,
        "ease":       600,
        "als":        1800,
    }
    TUNE_N_TRIALS = {
        "popularity": 10,
        "itemknn":    40,
        "ease":       15,
        "als":        50,
    }
    # Fold used as the Optuna objective ("a" = submission LOO, "b" = global LOO)
    TUNE_FOLD = "b"

    # ── Ensemble ─────────────────────────────────────────────────────
    ENSEMBLE_MODELS = ["ease", "als", "itemknn", "popularity"]
    ENSEMBLE_RRF_K = 60        # RRF constant (60 is the standard default)
