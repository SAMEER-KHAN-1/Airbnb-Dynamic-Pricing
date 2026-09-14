"""Generate a few exploratory charts from the cleaned listings data, for the viva."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PROCESSED_LISTINGS_PATH


def main() -> None:
    df = pd.read_csv(PROCESSED_LISTINGS_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. price distribution
    fig, ax = plt.subplots()
    df["price"].plot(kind="hist", bins=50, ax=ax)
    ax.set_title("Distribution of nightly price")
    ax.set_xlabel("Price ($)")
    fig.savefig(FIGURES_DIR / "price_distribution.png")
    plt.close(fig)

    # 2. price by room type
    fig, ax = plt.subplots()
    df.boxplot(column="price", by="room_type", ax=ax)
    ax.set_title("Price by room type")
    ax.set_ylabel("Price ($)")
    plt.suptitle("")
    fig.savefig(FIGURES_DIR / "price_by_room_type.png")
    plt.close(fig)

    # 3. average price by borough
    fig, ax = plt.subplots()
    df.groupby("neighbourhood_group_cleansed")["price"].mean().sort_values().plot(
        kind="barh", ax=ax
    )
    ax.set_title("Average price by borough")
    ax.set_xlabel("Average price ($)")
    fig.savefig(FIGURES_DIR / "price_by_borough.png")
    plt.close(fig)

    print(f"saved 3 charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
