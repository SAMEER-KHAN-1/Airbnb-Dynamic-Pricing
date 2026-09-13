import pandas as pd

# quick check on the reviews data - want to see if the same reviewer
# shows up more than once. if barely anyone does, collaborative filtering
# won't have much to learn from
reviews = pd.read_csv("Dataset/reviews.csv.gz")
print("total reviews:", len(reviews))

reviewer_counts = reviews["reviewer_id"].value_counts()
repeat_reviewers = reviewer_counts[reviewer_counts >= 2]

print("unique reviewers:", len(reviewer_counts))
print("reviewers with 2+ reviews:", len(repeat_reviewers))
print("% repeat reviewers:", round(len(repeat_reviewers) / len(reviewer_counts) * 100, 2))

# now check the listings file - size and how messy the price column is
listings = pd.read_csv("Dataset/listings.csv.gz", low_memory=False)

print()
print("total listings:", len(listings))
print("missing review_scores_rating:", listings["review_scores_rating"].isna().sum())
print("missing price:", listings["price"].isna().sum())
print("price sample:", listings["price"].head(10).tolist())
