"""Clean the raw Inside Airbnb listings export into a modeling-ready CSV.

Source: Inside Airbnb public dataset, NYC release (see Dataset/, not tracked
in git). Documented here since the PRD requires the data source and its
limitations to be explicit rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CATEGORICAL_FEATURES,
    LISTING_ID,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    PROCESSED_LISTINGS_PATH,
    RAW_LISTINGS_PATH,
    TARGET,
)


def clean_price(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .replace("nan", np.nan)
        .astype(float)
    )


def clean_bathrooms(df: pd.DataFrame) -> pd.Series:
    # "bathrooms" is often missing but "bathrooms_text" (e.g. "1.5 shared baths") usually isn't
    extracted = df["bathrooms_text"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    return df["bathrooms"].fillna(extracted.astype(float))


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_LISTINGS_PATH, low_memory=False)

    df[TARGET] = clean_price(df[TARGET])
    df["bathrooms"] = clean_bathrooms(df)

    keep_cols = [LISTING_ID] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df = df[keep_cols].copy()

    # drop duplicate listing ids and listings with no price at all
    df = df.drop_duplicates(subset=[LISTING_ID])
    df = df.dropna(subset=[TARGET])

    # a handful of listings are priced at $0 or absurdly high - drop outliers
    df = df[(df[TARGET] > 0) & (df[TARGET] < 2000)]

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df.to_csv(PROCESSED_LISTINGS_PATH, index=False)
    print(f"saved {len(df)} cleaned listings to {PROCESSED_LISTINGS_PATH}")


if __name__ == "__main__":
    main()
