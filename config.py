from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

RAW_LISTINGS_PATH = ROOT_DIR / "Dataset" / "listings.csv.gz"
RAW_REVIEWS_PATH = ROOT_DIR / "Dataset" / "reviews.csv.gz"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_LISTINGS_PATH = PROCESSED_DIR / "listings_clean.csv"
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.csv"

MODELS_DIR = ROOT_DIR / "models"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
CONTENT_MATCHER_PATH = MODELS_DIR / "content_matcher.joblib"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"

LISTING_ID = "id"

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
]

CATEGORICAL_FEATURES = [
    "room_type",
    "neighbourhood_group_cleansed",
]

TARGET = "price"

TOP_K = 10
