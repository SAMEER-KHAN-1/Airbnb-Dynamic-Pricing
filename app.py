"""Demo UI for the recommendation-based pricing workflow.

Note: the target application stack (per the PRD/ADR-005) is React Native +
FastAPI + PostgreSQL. This Streamlit app is a throwaway placeholder UI to
demo the working ML pipeline (SVD collaborative filtering + content-based
fallback + comparable-price aggregation) before that stack is built.
"""

import pandas as pd
import streamlit as st

from config import CONTENT_MATCHER_PATH, SVD_MODEL_PATH
from src import pricing_service

st.set_page_config(page_title="Airbnb Dynamic Pricing", page_icon="🏠")

st.title("Airbnb Dynamic Pricing Recommender")
st.caption(
    "Recommends a market-reference nightly price by finding comparable NYC "
    "listings and aggregating their observed prices. Not a guaranteed "
    "optimal price - see README for scope and limitations."
)

if not SVD_MODEL_PATH.exists() or not CONTENT_MATCHER_PATH.exists():
    st.error(
        "Models not found. Run, in order: `python src/data_prep.py`, "
        "`python src/interaction_matrix.py`, `python src/svd_recommender.py`, "
        "`python src/content_matcher.py`."
    )
    st.stop()


@st.cache_resource
def load_context():
    return pricing_service.load_pricing_context()


svd_algo, listings_df = load_context()

mode = st.radio(
    "What are you pricing?",
    ["An existing listing (by ID)", "A new listing (not yet listed)"],
    help="Existing listings use the SVD collaborative-filtering path. "
         "New listings have no review history, so they use the content-based "
         "cold-start path instead.",
)

if mode == "An existing listing (by ID)":
    sample_ids = listings_df["id"].sample(5, random_state=1).tolist()
    st.caption(f"Example listing IDs from the dataset: {sample_ids}")
    listing_id = st.number_input("Listing ID", min_value=0, value=int(sample_ids[0]), step=1)

    if st.button("Get recommended price", type="primary"):
        if listing_id not in listings_df["id"].values:
            st.error("That listing ID isn't in the cleaned dataset.")
        else:
            result = pricing_service.recommend_price_for_listing(
                svd_algo, listings_df, int(listing_id)
            )
            st.session_state["result"] = result

else:
    col1, col2 = st.columns(2)
    with col1:
        room_type = st.selectbox(
            "Room type", ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
        )
        neighbourhood_group = st.selectbox(
            "Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        )
        accommodates = st.slider("Accommodates (guests)", 1, 16, 2)
        bedrooms = st.slider("Bedrooms", 0, 8, 1)
    with col2:
        bathrooms = st.slider("Bathrooms", 0.0, 6.0, 1.0, step=0.5)
        minimum_nights = st.number_input("Minimum nights", min_value=1, value=2)
        number_of_reviews = st.number_input("Number of reviews", min_value=0, value=0)
        review_scores_rating = st.slider("Review score rating", 1.0, 5.0, 4.8, step=0.01)
    availability_365 = st.slider("Days available in next 365", 0, 365, 180)

    if st.button("Get recommended price", type="primary"):
        attributes = {
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "minimum_nights": minimum_nights,
            "availability_365": availability_365,
            "number_of_reviews": number_of_reviews,
            "review_scores_rating": review_scores_rating,
            "room_type": room_type,
            "neighbourhood_group_cleansed": neighbourhood_group,
        }
        result = pricing_service.recommend_price_for_new_listing(attributes, listings_df)
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Result")

    if result["insufficient_data"]:
        st.warning("Not enough comparable evidence to generate a recommendation.")
    else:
        st.metric("Recommended nightly price", f"${result['recommended_price']:.2f}")
        st.caption(
            "Cold-start / content-based fallback used"
            if result["fallback_used"]
            else "SVD collaborative-filtering comparables used"
        )

        st.write("Comparable listings this price is based on:")
        comparables_df = pd.DataFrame(result["comparables"]).rename(
            columns={"id": "listing_id", "price": "observed_price"}
        )
        st.table(comparables_df)
