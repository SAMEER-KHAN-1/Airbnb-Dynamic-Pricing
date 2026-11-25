"""
Unit tests for Pricing Aggregation Engine and LLM Context Builder.
Member 2 Responsibility: Automated Testing & Verification.
"""

import pytest
from src.pricing.pricing_engine import (
    clean_and_filter_prices,
    calculate_price_statistics,
    aggregate_prices,
    generate_price_recommendation,
)
from src.llm.llm_pipeline import LLMContextBuilder, HospitalityLLMEngine


def test_clean_and_filter_prices_removes_outliers():
    # Regular cluster around 100-150 with one extreme $2500 outlier
    raw_prices = [110.0, 120.0, 130.0, 140.0, 150.0, 2500.0]
    cleaned = clean_and_filter_prices(raw_prices)
    assert 2500.0 not in cleaned
    assert len(cleaned) == 5


def test_aggregate_prices_median_baseline():
    prices = [100.0, 120.0, 150.0, 180.0, 200.0]
    median_price = aggregate_prices(prices, method="median")
    assert median_price == 150.0


def test_generate_price_recommendation_standard():
    comparables = [
        {"price": 100.0, "similarity_score": 0.95},
        {"price": 120.0, "similarity_score": 0.90},
        {"price": 130.0, "similarity_score": 0.88},
        {"price": 140.0, "similarity_score": 0.85},
        {"price": 160.0, "similarity_score": 0.80},
    ]
    res = generate_price_recommendation(comparables)
    assert res["recommended_price"] == 130.0
    assert res["fallback_used"] is False
    assert res["comparable_count"] == 5


def test_generate_price_recommendation_triggers_cold_start_fallback():
    # Less than 5 comparables should trigger fallback_used=True
    comparables = [
        {"price": 110.0, "similarity_score": 0.85},
        {"price": 125.0, "similarity_score": 0.80},
    ]
    res = generate_price_recommendation(comparables, min_required_comparables=5)
    assert res["fallback_used"] is True
    assert res["comparable_count"] == 2


def test_llm_context_and_insight_generation():
    target = {
        "name": "Test Listing",
        "neighborhood": "Williamsburg",
        "bedrooms": 1,
        "accommodates": 2,
    }
    pricing_result = {
        "recommended_price": 145.0,
        "price_min": 110.0,
        "price_median": 145.0,
        "price_max": 190.0,
        "comparable_count": 8,
        "fallback_used": False,
        "aggregation_method": "median",
    }
    comps = [{"name": "Peer 1", "neighborhood": "Williamsburg", "price": 140.0, "similarity_score": 0.92}]

    context = LLMContextBuilder.build_context(target, pricing_result, comps)
    engine = HospitalityLLMEngine()
    insight = engine.generate_insight(context)

    assert insight.headline is not None
    assert len(insight.actionable_tips) > 0
    assert "Williamsburg" in insight.headline or "Williamsburg" in insight.pricing_rationale
