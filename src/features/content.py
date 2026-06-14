from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Text columns we join together for each item.
_TEXT_COLS = ["main_category", "title", "store", "categories",
              "features", "description", "subtitle"]


def build_content_tfidf(config, item_to_idx: dict, max_features: int = 30000,
                        min_df: int = 2):
    # Build a TF-IDF matrix in the same item order as the other models.
    from sklearn.feature_extraction.text import TfidfVectorizer

    meta = pd.read_csv(config.ITEM_META_PATH)

    # Put all available text for an item into one string.
    present = [c for c in _TEXT_COLS if c in meta.columns]
    meta["_text"] = (
        meta[present].fillna("").astype(str).agg(" ".join, axis=1)
    )

    # Look up text by item id.
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
    # TF-IDF normalizes rows by default.
    tfidf = vectorizer.fit_transform(docs)
    print(f"Content: TF-IDF matrix {tfidf.shape}, nnz={tfidf.nnz:,}")

    return tfidf.astype(np.float32)
