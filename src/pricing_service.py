"""Turns comparable listings into a recommended nightly price.

Two paths, matching the PRD's HLD:
  - an existing listing_id with review history -> SVD collaborative path
    (comparable listings from item latent-factor similarity)
  - a brand-new / cold-start listing description -> content-based path
    (comparable listings from attribute cosine similarity)

The recommended price is the median observed price of the retrieved
comparable listings - a robust aggregate, not a claim of an optimal price
(see the PRD's scope boundary).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import LISTING_ID, PROCESSED_LISTINGS_PATH, TARGET, TOP_K
from src import content_matcher, svd_recommender


def _prices_for(listings_df: pd.DataFrame, listing_ids: list[int]) -> pd.DataFrame:
    matches = listings_df[listings_df[LISTING_ID].isin(listing_ids)][[LISTING_ID, TARGET]]
    order = {lid: i for i, lid in enumerate(listing_ids)}
    return matches.sort_values(by=LISTING_ID, key=lambda s: s.map(order))


def recommend_price_for_listing(
    svd_algo, listings_df: pd.DataFrame, listing_id: int, k: int = TOP_K
) -> dict:
    """Existing listing -> try the SVD collaborative path first."""
    comparables = svd_recommender.get_comparable_listings(svd_algo, listing_id, k)

    if comparables is None:
        row = listings_df[listings_df[LISTING_ID] == listing_id].iloc[0].to_dict()
        return recommend_price_for_new_listing(row, listings_df, k=k)

    ids = [lid for lid, _ in comparables]
    prices = _prices_for(listings_df, ids)
    if prices.empty:
        return {
            "recommended_price": None,
            "comparables": [],
            "fallback_used": False,
            "insufficient_data": True,
        }

    return {
        "recommended_price": round(float(prices[TARGET].median()), 2),
        "comparables": prices.to_dict("records"),
        "fallback_used": False,
        "insufficient_data": False,
    }


def recommend_price_for_new_listing(
    attributes: dict, listings_df: pd.DataFrame, k: int = TOP_K
) -> dict:
    """Cold-start / new listing -> content-based comparable retrieval."""
    matcher_state = content_matcher.load_content_matcher()
    similar = content_matcher.top_k_similar(matcher_state, attributes, k)

    ids = [lid for lid, _ in similar]
    prices = _prices_for(listings_df, ids)
    if prices.empty:
        return {
            "recommended_price": None,
            "comparables": [],
            "fallback_used": True,
            "insufficient_data": True,
        }

    return {
        "recommended_price": round(float(prices[TARGET].median()), 2),
        "comparables": prices.to_dict("records"),
        "fallback_used": True,
        "insufficient_data": False,
    }


def load_pricing_context():
    svd_algo = svd_recommender.load_model()
    listings_df = pd.read_csv(PROCESSED_LISTINGS_PATH)
    return svd_algo, listings_df
