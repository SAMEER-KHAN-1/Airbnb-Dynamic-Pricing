"""Build a user-item interaction table for collaborative filtering.

Data source note (documented per PRD data-quality requirements): this Inside
Airbnb release has no explicit star ratings. reviews.csv.gz gives us
(reviewer_id, listing_id) pairs - i.e. implicit feedback ("this reviewer
stayed at and reviewed this listing"), not an explicit preference score.

We do not fabricate ratings. Instead we use how many times a reviewer
reviewed a given listing as an implicit interaction strength, capped at 5 so
it behaves like a bounded rating scale for Surprise's SVD. This is a
documented limitation, not a substitute for real explicit ratings.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    INTERACTIONS_PATH,
    LISTING_ID,
    PROCESSED_DIR,
    PROCESSED_LISTINGS_PATH,
    RAW_REVIEWS_PATH,
)

MAX_IMPLICIT_STRENGTH = 5


def build_interactions() -> pd.DataFrame:
    reviews = pd.read_csv(RAW_REVIEWS_PATH, usecols=["listing_id", "reviewer_id"])
    listings = pd.read_csv(PROCESSED_LISTINGS_PATH, usecols=[LISTING_ID])

    # only keep interactions for listings that survived cleaning
    reviews = reviews[reviews["listing_id"].isin(listings[LISTING_ID])]

    interactions = (
        reviews.groupby(["reviewer_id", "listing_id"])
        .size()
        .clip(upper=MAX_IMPLICIT_STRENGTH)
        .reset_index(name="strength")
    )
    return interactions


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    interactions = build_interactions()
    interactions.to_csv(INTERACTIONS_PATH, index=False)
    print(f"saved {len(interactions)} interactions to {INTERACTIONS_PATH}")
    print(f"  unique reviewers: {interactions['reviewer_id'].nunique()}")
    print(f"  unique listings: {interactions['listing_id'].nunique()}")


if __name__ == "__main__":
    main()
