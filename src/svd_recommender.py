"""SVD collaborative filtering over the reviewer-listing interaction matrix.

Primary recommendation model per the PRD. Because interactions here are
implicit (see src/interaction_matrix.py) rather than explicit star ratings,
RMSE and Precision@10 are reported as documented, simplified adaptations of
those metrics to implicit-only data - not a substitute for a dataset with
real explicit ratings. This limitation is called out deliberately rather
than presenting the numbers as if they meant the same thing they would for
an explicit-rating recommender (e.g. MovieLens).
"""

import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import INTERACTIONS_PATH, SVD_MODEL_PATH

RATING_SCALE = (1, 5)
RELEVANCE_THRESHOLD = 1  # any observed interaction counts as a positive signal


def load_dataset() -> Dataset:
    interactions = pd.read_csv(INTERACTIONS_PATH)
    reader = Reader(rating_scale=RATING_SCALE)
    return Dataset.load_from_df(
        interactions[["reviewer_id", "listing_id", "strength"]], reader
    )


def precision_recall_at_k(predictions, k=10, threshold=RELEVANCE_THRESHOLD):
    """Standard Surprise-FAQ precision/recall@k, adapted to implicit data
    (see module docstring for the caveat)."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in ratings)
        n_rec_k = sum(est >= threshold for est, _ in ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold and est >= threshold) for est, true_r in ratings[:k]
        )
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall


def train_and_evaluate() -> tuple:
    data = load_dataset()
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=10)

    metrics = {"rmse": rmse, "precision_at_10": precision, "recall_at_10": recall}
    return algo, metrics


def save_model(algo) -> None:
    joblib.dump(algo, SVD_MODEL_PATH)


def load_model():
    return joblib.load(SVD_MODEL_PATH)


def get_comparable_listings(algo, listing_id: int, k: int) -> list[tuple[int, float]] | None:
    """Top-k comparable listings by cosine similarity of SVD item latent
    factors. Returns None if listing_id has no interaction history (cold
    start - caller should fall back to the content-based matcher)."""
    trainset = algo.trainset
    try:
        inner_id = trainset.to_inner_iid(listing_id)
    except ValueError:
        return None

    target_vector = algo.qi[inner_id].reshape(1, -1)
    sims = cosine_similarity(target_vector, algo.qi)[0]

    ranked = sorted(
        ((idx, sim) for idx, sim in enumerate(sims) if idx != inner_id),
        key=lambda pair: pair[1],
        reverse=True,
    )[:k]
    return [(trainset.to_raw_iid(idx), float(sim)) for idx, sim in ranked]


def main() -> None:
    algo, metrics = train_and_evaluate()
    save_model(algo)
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"Precision@10: {metrics['precision_at_10']:.3f}")
    print(f"Recall@10: {metrics['recall_at_10']:.3f}")
    print(f"saved SVD model to {SVD_MODEL_PATH}")


if __name__ == "__main__":
    main()
