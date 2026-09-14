"""End-to-end smoke test: processed data, interactions, SVD model and content
matcher all exist and produce a sane recommendation via both the SVD path
(existing listing) and the content-based cold-start path (new listing)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CONTENT_MATCHER_PATH,
    INTERACTIONS_PATH,
    PROCESSED_LISTINGS_PATH,
    SVD_MODEL_PATH,
)
from src import pricing_service


def main() -> None:
    for path, hint in [
        (PROCESSED_LISTINGS_PATH, "run src/data_prep.py first"),
        (INTERACTIONS_PATH, "run src/interaction_matrix.py first"),
        (SVD_MODEL_PATH, "run src/svd_recommender.py first"),
        (CONTENT_MATCHER_PATH, "run src/content_matcher.py first"),
    ]:
        assert path.exists(), hint

    svd_algo, listings_df = pricing_service.load_pricing_context()
    assert len(listings_df) > 0, "processed dataset is empty"

    existing_id = int(listings_df.iloc[0]["id"])
    existing_result = pricing_service.recommend_price_for_listing(
        svd_algo, listings_df, existing_id
    )
    assert not existing_result["insufficient_data"], "SVD path returned no comparables"
    assert existing_result["recommended_price"] > 0

    new_listing_result = pricing_service.recommend_price_for_new_listing(
        {
            "accommodates": 4, "bedrooms": 2, "bathrooms": 1.0, "minimum_nights": 3,
            "availability_365": 200, "number_of_reviews": 0, "review_scores_rating": 4.8,
            "room_type": "Entire home/apt", "neighbourhood_group_cleansed": "Brooklyn",
        },
        listings_df,
    )
    assert not new_listing_result["insufficient_data"], "content-based path returned no comparables"
    assert new_listing_result["fallback_used"] is True
    assert new_listing_result["recommended_price"] > 0

    print("all checks passed")
    print(f"  processed listings: {len(listings_df)}")
    print(f"  SVD path (listing {existing_id}): ${existing_result['recommended_price']:.2f}")
    print(f"  content-based path (new listing): ${new_listing_result['recommended_price']:.2f}")


if __name__ == "__main__":
    main()
