"""Content-based comparable-listing retrieval using cosine similarity.

Used as the cold-start path: for a brand-new listing description that has no
review history (and therefore isn't in the SVD's trained item set), we find
similar existing listings by their attributes instead.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CATEGORICAL_FEATURES,
    CONTENT_MATCHER_PATH,
    LISTING_ID,
    NUMERIC_FEATURES,
    PROCESSED_LISTINGS_PATH,
)


def build_encoder() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def fit_content_matcher(df: pd.DataFrame) -> dict:
    encoder = build_encoder()
    feature_matrix = encoder.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    feature_matrix = np.asarray(feature_matrix.todense()) if hasattr(feature_matrix, "todense") else feature_matrix
    return {
        "encoder": encoder,
        "feature_matrix": feature_matrix,
        "listing_ids": df[LISTING_ID].to_numpy(),
    }


def save_content_matcher(state: dict) -> None:
    joblib.dump(state, CONTENT_MATCHER_PATH)


def load_content_matcher() -> dict:
    return joblib.load(CONTENT_MATCHER_PATH)


def top_k_similar(state: dict, attributes: dict, k: int) -> list[tuple[int, float]]:
    """attributes: dict of the NUMERIC_FEATURES + CATEGORICAL_FEATURES for a listing
    (new or existing). Returns [(listing_id, similarity), ...] sorted descending."""
    row = pd.DataFrame([attributes])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    vector = state["encoder"].transform(row)
    vector = np.asarray(vector.todense()) if hasattr(vector, "todense") else vector

    sims = cosine_similarity(vector, state["feature_matrix"])[0]
    top_idx = np.argsort(-sims)[:k]
    return [(int(state["listing_ids"][i]), float(sims[i])) for i in top_idx]


def main() -> None:
    df = pd.read_csv(PROCESSED_LISTINGS_PATH)
    state = fit_content_matcher(df)
    save_content_matcher(state)
    print(f"saved content matcher ({state['feature_matrix'].shape[1]} features, "
          f"{len(state['listing_ids'])} listings) to {CONTENT_MATCHER_PATH}")


if __name__ == "__main__":
    main()
