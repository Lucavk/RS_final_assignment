"""
Item content features from item_meta.csv.

Builds a TF-IDF matrix over concatenated text fields (title, categories,
features, description, brand, …).  All features are derived ONLY from the
provided data — no external pretrained embeddings (competition rule).

Output rows are aligned to the model's item index space (item_to_idx);
items with no metadata get an all-zero row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Text columns to concatenate into one "document" per item.
_TEXT_COLS = ["main_category", "title", "store", "categories",
              "features", "description", "subtitle"]


def build_content_tfidf(config, item_to_idx: dict, max_features: int = 30000,
                        min_df: int = 2):
    """
    Build an L2-normalised TF-IDF matrix aligned to item_to_idx.

    Returns
    -------
    tfidf : csr_matrix [n_items × n_features]
        Row i corresponds to the item with index i in item_to_idx.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    meta = pd.read_csv(config.ITEM_META_PATH)

    # Concatenate available text columns into a single string per row
    present = [c for c in _TEXT_COLS if c in meta.columns]
    meta["_text"] = (
        meta[present].fillna("").astype(str).agg(" ".join, axis=1)
    )

    # Map item_id -> document text
    id_to_text = dict(zip(meta["item_id"], meta["_text"]))

    n_items = len(item_to_idx)
    docs = [""] * n_items
    n_missing = 0
    for item_id, idx in item_to_idx.items():
        text = id_to_text.get(item_id)
        if text is None:
            n_missing += 1
            text = ""
        docs[idx] = text

    print(f"Content: {n_items - n_missing}/{n_items} items have metadata "
          f"({n_missing} missing → zero rows)")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(docs)   # already L2-normalised by default
    print(f"Content: TF-IDF matrix {tfidf.shape}, nnz={tfidf.nnz:,}")

    return tfidf.astype(np.float32)
