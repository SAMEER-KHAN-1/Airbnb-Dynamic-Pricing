"""
LLM Intelligence Pipeline for Airbnb Dynamic Pricing Recommender.
Member 2 Responsibility: Context Builder, Prompt Engineering, Structured LLM Output, and Validation.
Supports Local Neural LLM (Ollama / Llama 3.2) with graceful fallback to local intelligence.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import requests
import os


class PricingInsightSchema(BaseModel):
    """Structured Pydantic schema enforced on LLM output."""
    headline: str = Field(description="One-sentence executive summary of the price recommendation.")
    market_tier: str = Field(description="Competitive position within the neighborhood (e.g., budget, mid-market, premium).")
    pricing_rationale: str = Field(description="Why this specific price is recommended based on peer data.")
    actionable_tips: List[str] = Field(description="Actionable pricing or amenity strategies for the host.")
    comparable_summary: str = Field(description="Summary of the peer listings used for comparison.")
    confidence_commentary: str = Field(description="Assessment of recommendation reliability and fallback status.")


class LLMContextBuilder:
    """Builds a compact, structured JSON payload for the LLM from ML results."""

    @staticmethod
    def build_context(
        target_listing: Dict[str, Any],
        pricing_result: Dict[str, Any],
        comparables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "target_listing": {
                "name": target_listing.get("name", "Listing"),
                "neighborhood": target_listing.get("neighborhood", "Unknown"),
                "property_type": target_listing.get("property_type", "Entire rental unit"),
                "room_type": target_listing.get("room_type", "Entire home/apt"),
                "accommodates": target_listing.get("accommodates", 2),
                "bedrooms": target_listing.get("bedrooms", 1),
                "bathrooms": target_listing.get("bathrooms", 1.0),
                "review_count": target_listing.get("review_count", 0),
                "review_rating": target_listing.get("review_rating"),
            },
            "pricing_metrics": {
                "recommended_price": pricing_result.get("recommended_price", 0.0),
                "price_min": pricing_result.get("price_min", 0.0),
                "price_median": pricing_result.get("price_median", 0.0),
                "price_max": pricing_result.get("price_max", 0.0),
                "price_mean": pricing_result.get("price_mean", 0.0),
                "comparable_count": pricing_result.get("comparable_count", 0),
                "fallback_used": pricing_result.get("fallback_used", False),
                "aggregation_method": pricing_result.get("aggregation_method", "median"),
            },
            "top_comparables": [
                {
                    "name": c.get("name"),
                    "neighborhood": c.get("neighborhood"),
                    "price": c.get("price"),
                    "similarity_score": round(c.get("similarity_score", 1.0), 2),
                }
                for c in comparables[:5]
            ],
        }


class HospitalityLLMEngine:
    """
    Hospitality Revenue Management LLM Engine.
    Attempts to call local Ollama (Llama 3.2), falling back gracefully to local intelligence.
    """

    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    DEFAULT_MODEL = "llama3.2:1b"

    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name

    def _query_local_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Queries local Ollama instance running on the laptop."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                },
            }
            res = requests.post(self.OLLAMA_URL, json=payload, timeout=20)
            if res.status_code == 200:
                raw_text = res.json().get("response", "")
                return json.loads(raw_text)
        except Exception:
            return None
        return None

    def generate_insight(self, context: Dict[str, Any]) -> PricingInsightSchema:
        """
        Executes the LLM pipeline and returns a validated Pydantic model.
        """
        target = context["target_listing"]
        metrics = context["pricing_metrics"]
        comps = context.get("top_comparables", [])

        # Try querying local Ollama first
        prompt = (
            f"You are an expert Hospitality Revenue Strategist for short-term rentals.\n"
            f"Analyze this Airbnb listing and market peer pricing data:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            f"Respond strictly in valid JSON matching this structure:\n"
            f'{{\n'
            f'  "headline": "one sentence summary",\n'
            f'  "market_tier": "e.g. Budget, Mid-Market, or Premium",\n'
            f'  "pricing_rationale": "explanation of price relative to peers",\n'
            f'  "actionable_tips": ["tip 1", "tip 2", "tip 3"],\n'
            f'  "comparable_summary": "summary of the peers used",\n'
            f'  "confidence_commentary": "reliability note"\n'
            f'}}\n'
        )

        ollama_result = self._query_local_ollama(prompt)
        if ollama_result:
            try:
                # Ensure all required keys exist and validate
                return PricingInsightSchema(
                    headline=ollama_result.get("headline", f"Recommended at ${metrics['recommended_price']:.0f}/night."),
                    market_tier=ollama_result.get("market_tier", "Competitive Market Tier"),
                    pricing_rationale=ollama_result.get("pricing_rationale", "Peer-aligned rate."),
                    actionable_tips=ollama_result.get("actionable_tips", ["Optimize weekend pricing."]),
                    comparable_summary=ollama_result.get("comparable_summary", f"Based on {metrics['comparable_count']} peers."),
                    confidence_commentary=f"Generated live by local {self.model_name} on device. " + ollama_result.get("confidence_commentary", ""),
                )
            except Exception:
                pass  # Fallback to local rule engine if parsing fails

        # Fallback Local Domain Engine (Guarantees zero-failure demo)
        if metrics.get("fallback_used"):
            return PricingInsightSchema(
                headline=f"Recommended at ${metrics['recommended_price']:.0f}/night based on {metrics['comparable_count']} physical attribute matches in {target['neighborhood']}.",
                market_tier="Cold-Start / Introductory Tier",
                pricing_rationale=f"This listing has limited or no historical review data. The pricing engine activated the Content-Based Fallback to identify {metrics['comparable_count']} comparable units with similar room types and capacities (${metrics['price_min']:.0f} - ${metrics['price_max']:.0f}).",
                actionable_tips=[
                    "Offer an initial 15-20% 'New Listing' promotional discount for the first 3 bookings to gather verified 5-star reviews quickly.",
                    "Highlight high-demand amenities (fast Wi-Fi, self check-in) in photos to build immediate guest trust.",
                    "Review pricing once the property achieves its first 5 guest reviews.",
                ],
                comparable_summary=f"Compared against {len(comps)} physically similar units in {target['neighborhood']}. Observed rates range from ${metrics['price_min']:.0f} to ${metrics['price_max']:.0f}.",
                confidence_commentary="Moderate confidence (Fallback mode active). Switched from Collaborative Filtering to Content-Based Similarity due to lack of historical booking interactions.",
            )

        name = target.get("name", "Listing")
        rec_price = metrics["recommended_price"]
        p_min = metrics["price_min"]
        p_max = metrics["price_max"]
        p_med = metrics["price_median"]
        n_comps = metrics["comparable_count"]
        hood = target.get("neighborhood", "the local area")

        if rec_price >= p_med * 1.1:
            tier = "Upper Mid-Market / Premium Tier"
        elif rec_price <= p_med * 0.9:
            tier = "Value / Competitive Tier"
        else:
            tier = "Market Median Tier"

        return PricingInsightSchema(
            headline=f"Recommended at ${rec_price:.0f}/night based on {n_comps} peer listings in {hood}.",
            market_tier=tier,
            pricing_rationale=(
                f"The target listing aligns with {n_comps} high-similarity market peers whose nightly rates center at a median of ${p_med:.0f} "
                f"(ranging between ${p_min:.0f} and ${p_max:.0f}). The {target.get('bedrooms', 1)}BR layout and capacity of {target.get('accommodates', 2)} guests "
                f"position it squarely in active traveler demand."
            ),
            actionable_tips=[
                "Implement dynamic weekend premiums: add a +10% to +15% surge for Friday and Saturday night stays.",
                "Maintain a 2-night minimum stay during peak periods to minimize turnover and cleaning costs.",
                "Ensure key amenities such as dedicated workspace and coffee bar are clearly tagged in listing titles to justify rates above the median.",
            ],
            comparable_summary=f"Matched with {n_comps} peer listings with similarity scores up to {comps[0]['similarity_score'] if comps else 0.9} in {hood}.",
            confidence_commentary="High confidence. Recommendation is derived from robust collaborative filtering review patterns across active market peers.",
        )
