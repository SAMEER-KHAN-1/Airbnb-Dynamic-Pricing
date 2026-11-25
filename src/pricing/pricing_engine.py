"""
Pricing Aggregation and Fallback Engine for Airbnb Dynamic Pricing Recommender.
Member 2 Responsibility: Pricing Layer.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def clean_and_filter_prices(
    prices: List[float],
    min_valid: float = 10.0,
    max_valid: float = 5000.0,
    iqr_multiplier: float = 1.5,
) -> List[float]:
    """
    Cleans raw observed prices, removes NaNs/negatives, and filters extreme outliers using IQR.
    """
    valid = [float(p) for p in prices if p is not None and not np.isnan(p) and min_valid <= p <= max_valid]
    if len(valid) < 4:
        return valid

    # Interquartile Range (IQR) filtering
    q25, q75 = np.percentile(valid, [25, 75])
    iqr = q75 - q25
    lower_bound = max(min_valid, q25 - (iqr_multiplier * iqr))
    upper_bound = min(max_valid, q75 + (iqr_multiplier * iqr))

    filtered = [p for p in valid if lower_bound <= p <= upper_bound]
    # If IQR was too aggressive, return original valid prices
    return filtered if len(filtered) >= 2 else valid


def calculate_price_statistics(prices: List[float]) -> Dict[str, float]:
    """
    Computes statistical bounds (min, 25th percentile, median, 75th percentile, max, mean).
    """
    if not prices:
        return {
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }

    arr = np.array(prices)
    return {
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.round(np.mean(arr), 2)),
        "std": float(np.round(np.std(arr), 2)),
    }


def aggregate_prices(
    prices: List[float],
    weights: Optional[List[float]] = None,
    method: str = "median",
) -> float:
    """
    Aggregates observed prices into a single recommended nightly rate.
    Supported methods: 'median' (robust baseline), 'mean', 'weighted_mean'.
    """
    if not prices:
        return 0.0

    if method == "weighted_mean" and weights and len(weights) == len(prices):
        weight_sum = sum(weights)
        if weight_sum > 0:
            return float(np.round(np.average(prices, weights=weights), 2))

    if method == "mean":
        return float(np.round(np.mean(prices), 2))

    # Default: Median (defensible baseline against market skew)
    return float(np.round(np.median(prices), 2))


def generate_price_recommendation(
    comparables: List[Dict[str, Any]],
    min_required_comparables: int = 5,
    preferred_method: str = "median",
) -> Dict[str, Any]:
    """
    Given a list of comparable listings with observed prices and similarity scores,
    filters the prices, computes statistics, checks fallback conditions,
    and returns a structured pricing result.
    """
    raw_prices = [item.get("price") for item in comparables if item.get("price") is not None]
    weights = [item.get("similarity_score", 1.0) for item in comparables if item.get("price") is not None]

    cleaned_prices = clean_and_filter_prices(raw_prices)
    stats = calculate_price_statistics(cleaned_prices)

    fallback_used = len(cleaned_prices) < min_required_comparables

    # If fallback is needed or weights are unavailable, use median
    recommended_price = aggregate_prices(
        cleaned_prices,
        weights=weights if not fallback_used else None,
        method=preferred_method if not fallback_used else "median",
    )

    return {
        "recommended_price": recommended_price,
        "comparable_count": len(cleaned_prices),
        "raw_comparable_count": len(raw_prices),
        "price_min": stats["min"],
        "price_q25": stats["q25"],
        "price_median": stats["median"],
        "price_q75": stats["q75"],
        "price_max": stats["max"],
        "price_mean": stats["mean"],
        "fallback_used": fallback_used,
        "aggregation_method": preferred_method if not fallback_used else "median_fallback",
    }
