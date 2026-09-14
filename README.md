# airbnb-dynamic-pricing-recommender

A recommendation-system approach to Airbnb pricing: identify comparable
listings and use their observed prices as a market-reference price
recommendation. Not Airbnb's internal pricing engine and not a guarantee of
an optimal price — see Scope below.

Built on the NYC Inside Airbnb export (`Dataset/listings.csv.gz`,
`Dataset/reviews.csv.gz`; not tracked in git — see `.gitignore`).

## Target architecture vs. current state

The target stack is **React Native → FastAPI → PostgreSQL**, per the
project's PRD. This is a large, multi-week build. What exists right now is
the **data science core** (recommendation pipeline) plus a **throwaway
Streamlit UI** used only to demo that pipeline while the real application
layer is being built. `app.py` will be replaced by the React Native +
FastAPI + PostgreSQL stack in a later milestone.

## How the recommendation pipeline works

1. **`src/data_prep.py`** — cleans `listings.csv.gz`: parses price, fills
   missing numeric/categorical fields, drops listings with no price or
   extreme outliers. Writes `data/processed/listings_clean.csv`.
2. **`src/interaction_matrix.py`** — builds a reviewer-listing interaction
   table from `reviews.csv.gz`. **Data limitation:** this dataset release has
   no explicit star ratings, only (reviewer, listing) review pairs. We use
   how many times a reviewer reviewed a given listing as an implicit
   interaction strength (capped at 5) rather than fabricating ratings. This
   is documented, not hidden.
3. **`src/svd_recommender.py`** — trains `Surprise`'s SVD collaborative
   filtering model on that interaction matrix (80/20 split). Comparable
   listings for an existing listing are retrieved via cosine similarity of
   the model's learned item latent factors.
4. **`src/content_matcher.py`** — content-based cosine similarity over
   listing attributes (room type, borough, capacity, etc). Used as the
   cold-start path for listings with no review history.
5. **`src/pricing_service.py`** — orchestrates both paths: existing listings
   go through SVD comparables; new/cold-start listings fall back to
   content-based comparables. The recommended price is the **median**
   observed price of the top-K comparable listings.
6. **`app.py`** — Streamlit demo UI over `pricing_service` (temporary, see
   above).

## Evaluation

- **RMSE** — how well the SVD model reconstructs held-out interaction
  strengths.
- **Precision@10 / Recall@10** — computed with the standard formula, but
  with a caveat worth understanding: because this dataset only has
  *positive* implicit interactions (no negative examples), both come out
  near 1.0 — a known limitation of applying explicit-rating precision/recall
  to implicit-only data without negative sampling or full-catalog ranking
  evaluation. RMSE is the more informative number here.

Run `python src/svd_recommender.py` to print both.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running the pipeline

Run these in order (only needed once, or whenever `Dataset/` changes):

```bash
python src/data_prep.py           # Dataset/ -> data/processed/listings_clean.csv
python src/interaction_matrix.py  # -> data/processed/interactions.csv
python src/svd_recommender.py     # trains SVD, prints RMSE/Precision@10, saves models/svd_model.pkl
python src/content_matcher.py     # -> models/content_matcher.joblib
python scripts/sanity_check.py    # optional: verify the pipeline end-to-end
python scripts/eda.py             # optional: regenerate charts in reports/figures/
```

## Running the demo UI

```bash
streamlit run app.py
```

## Scope boundary

Recommends a market-reference nightly price from public historical Airbnb
data. It is not Airbnb's internal pricing engine, does not guarantee optimal
pricing or revenue uplift, does not autonomously change prices, and does not
use private Airbnb operational data. Historical observed price is a proxy
signal, not ground truth for an economically optimal price.

## Project status

Data pipeline + recommendation core (SVD collaborative filtering, content-
based fallback, comparable-price aggregation) are working end-to-end.
Not yet built: FastAPI backend, PostgreSQL persistence, React Native app,
hybrid blending, automated tests, CI/CD, deployment.
