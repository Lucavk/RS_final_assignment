import os
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
    SCORES_DIR = Path(os.environ.get("RS_SCORES_DIR", str(ARTIFACTS_DIR / "scores")))
    PARAMS_DIR = ARTIFACTS_DIR / "params"
    OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
    MODELS_DIR = ARTIFACTS_DIR / "models"

    K = 10

    # Popularity.
    # Half-life in days for exponential time decay.
    POPULARITY_HALFLIFE_DAYS = 365

    # Item-item cosine KNN
    ITEMKNN_TOPK = 200
    ITEMKNN_SHRINKAGE = 100

    # EASE
    EASE_LAMBDA = 500.0

    # ALS
    ALS_FACTORS = 128
    ALS_REGULARIZATION = 0.05
    ALS_ITERATIONS = 50
    ALS_ALPHA = 20.0

    # BPR
    BPR_FACTORS = 128
    BPR_LEARNING_RATE = 0.01
    BPR_REGULARIZATION = 0.01
    BPR_ITERATIONS = 100

    # Mult-VAE
    VAE_HIDDEN = 600
    VAE_LATENT = 200
    VAE_DROPOUT = 0.5
    VAE_LR = 1e-3
    VAE_WEIGHT_DECAY = 0.0
    VAE_BATCH_SIZE = 500
    VAE_EPOCHS = 150
    VAE_BETA = 0.2
    VAE_ANNEAL_EPOCHS = 50

    # Content-KNN
    CONTENT_TOPK = 200
    CONTENT_MAX_FEATURES = 30000
    CONTENT_MIN_DF = 2

    # LightGCN
    LIGHTGCN_DIM = 128
    LIGHTGCN_LAYERS = 2
    LIGHTGCN_EPOCHS = 30
    LIGHTGCN_BATCH_SIZE = 2048
    LIGHTGCN_LR = 1e-3
    LIGHTGCN_WEIGHT_DECAY = 1e-6

    # BERT4Rec
    BERT4REC_MAX_SEQ_LEN = 20
    BERT4REC_DIM = 128
    BERT4REC_HEADS = 4
    BERT4REC_LAYERS = 2
    BERT4REC_DROPOUT = 0.2
    BERT4REC_EPOCHS = 20
    BERT4REC_BATCH_SIZE = 512
    BERT4REC_LR = 1e-3

    # Recency transitions
    RECENCY_WINDOW = 3
    RECENCY_DECAY = 1.0
    RECENCY_RECENCY_DECAY = 0.5
    RECENCY_SHRINKAGE = 10.0
    RECENCY_POP_DISCOUNT = 0.0
    RECENCY_MAX_RECENT = 20

    # Tuning.
    # Optuna time limits per model (seconds)
    TUNE_TIMEOUT = {
        "popularity": 120,
        "itemknn":    1200,
        "ease":       600,
        "als":        1800,
        "bpr":        1800,
        "multvae":    3600,
        "content":    1200,
        "lightgcn":   3600,
        "bert4rec":   3600,
        "recency":    1800,
    }
    TUNE_N_TRIALS = {
        "popularity": 10,
        "itemknn":    40,
        "ease":       20,
        "als":        50,
        "bpr":        40,
        "multvae":    20,
        "content":    10,
        "lightgcn":   15,
        "bert4rec":   12,
        "recency":    40,
    }
    # Fold used as the Optuna objective.
    TUNE_FOLD = "b"

    # Ensemble.
    # All trained models; subset via --models on the CLI
    ENSEMBLE_MODELS = ["ease", "als", "itemknn", "popularity",
                       "bpr", "multvae", "content", "lightgcn", "bert4rec",
                       "recency"]
    ENSEMBLE_RRF_K = 60
    # Fold for ensemble weight tuning. "b" = robust (many users), "a" = faithful
    ENSEMBLE_TUNE_FOLD = "b"
