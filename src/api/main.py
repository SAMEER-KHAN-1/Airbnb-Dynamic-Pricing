"""
FastAPI Server for Airbnb Dynamic Pricing Recommender.
Member 2 Responsibility: API Layer, Pipeline Orchestration, and Response Contract.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid

from src.pricing.pricing_engine import generate_price_recommendation
from src.pricing.mock_provider import (
    SAMPLE_TARGET_LISTINGS,
    get_target_listing,
    get_mock_comparables,
)
from src.llm.llm_pipeline import (
    LLMContextBuilder,
    HospitalityLLMEngine,
    PricingInsightSchema,
)

app = FastAPI(
    title="Airbnb Dynamic Pricing Recommender API",
    description="Machine-learning market-reference pricing and AI explanation engine.",
    version="1.0.0",
)

# Enable CORS for React Native / mobile web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_engine = HospitalityLLMEngine()


# Pydantic Request Models
class RecommendationRequest(BaseModel):
    listing_id: Optional[int] = Field(default=101, description="ID of target listing from dataset or mock store")
    mode: str = Field(
        default="pricing_with_insight",
        description="Execution mode: 'pricing' (ML only), 'insight' (LLM only), or 'pricing_with_insight' (Combined)",
    )
    custom_listing: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom listing inputs")


# Pydantic Response Models
class RecommendationResponse(BaseModel):
    recommendation_id: str
    listing_id: Optional[int]
    target_listing: Dict[str, Any]
    currency: str = "USD"
    mode: str
    recommended_price: Optional[float] = None
    price_min: Optional[float] = None
    price_median: Optional[float] = None
    price_max: Optional[float] = None
    price_mean: Optional[float] = None
    comparable_count: Optional[int] = None
    fallback_used: Optional[bool] = None
    model_version: str
    insight: Optional[PricingInsightSchema] = None
    comparables: Optional[List[Dict[str, Any]]] = None


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint for API and pipeline readiness."""
    return {
        "status": "healthy",
        "service": "Airbnb Dynamic Pricing Recommender",
        "active_models": ["svd-collaborative-v1", "content-fallback-v1", "llm-revenue-advisor"],
        "pipeline_modes": ["pricing", "insight", "pricing_with_insight"],
    }


@app.get("/api/v1/listings")
def list_sample_listings():
    """Returns sample target listings available for demo testing."""
    return list(SAMPLE_TARGET_LISTINGS.values())


@app.get("/api/v1/listings/{listing_id}")
def get_listing_detail(listing_id: int):
    """Retrieve detailed metadata for a specific listing."""
    if listing_id not in SAMPLE_TARGET_LISTINGS:
        raise HTTPException(status_code=404, detail=f"Listing ID {listing_id} not found.")
    return SAMPLE_TARGET_LISTINGS[listing_id]


@app.post("/api/v1/recommendations/price", response_model=RecommendationResponse)
def recommend_price(req: RecommendationRequest):
    """
    Main Pricing & Insight endpoint.
    Orchestrates the Pricing Aggregation Engine and LLM Intelligence Pipeline.
    Supports 3 modes:
      1. 'pricing': Returns numerical price recommendation only.
      2. 'insight': Returns LLM structured analysis only.
      3. 'pricing_with_insight': End-to-end price computation + LLM explanation.
    """
    rec_id = f"rec_{uuid.uuid4().hex[:8]}"

    # 1. Resolve Target Listing
    if req.custom_listing:
        target = req.custom_listing
        listing_id = req.custom_listing.get("id", 999)
    else:
        listing_id = req.listing_id or 101
        target = get_target_listing(listing_id)

    # 2. Retrieve Comparable Listings (from SVD or Content Fallback)
    comparables = get_mock_comparables(listing_id)

    # 3. Mode: Pricing only
    if req.mode == "pricing":
        pricing_result = generate_price_recommendation(comparables)
        return RecommendationResponse(
            recommendation_id=rec_id,
            listing_id=listing_id,
            target_listing=target,
            mode="pricing",
            recommended_price=pricing_result["recommended_price"],
            price_min=pricing_result["price_min"],
            price_median=pricing_result["price_median"],
            price_max=pricing_result["price_max"],
            price_mean=pricing_result["price_mean"],
            comparable_count=pricing_result["comparable_count"],
            fallback_used=pricing_result["fallback_used"],
            model_version="svd-collaborative-v1" if not pricing_result["fallback_used"] else "content-fallback-v1",
            comparables=comparables,
        )

    # 4. Mode: Insight only (assumes baseline pricing already exists)
    if req.mode == "insight":
        pricing_result = generate_price_recommendation(comparables)
        context = LLMContextBuilder.build_context(target, pricing_result, comparables)
        insight = llm_engine.generate_insight(context)
        return RecommendationResponse(
            recommendation_id=rec_id,
            listing_id=listing_id,
            target_listing=target,
            mode="insight",
            model_version="llm-insight-v1",
            insight=insight,
        )

    # 5. Mode: Pricing with Insight (Combined end-to-end)
    pricing_result = generate_price_recommendation(comparables)
    context = LLMContextBuilder.build_context(target, pricing_result, comparables)
    insight = llm_engine.generate_insight(context)

    return RecommendationResponse(
        recommendation_id=rec_id,
        listing_id=listing_id,
        target_listing=target,
        mode="pricing_with_insight",
        recommended_price=pricing_result["recommended_price"],
        price_min=pricing_result["price_min"],
        price_median=pricing_result["price_median"],
        price_max=pricing_result["price_max"],
        price_mean=pricing_result["price_mean"],
        comparable_count=pricing_result["comparable_count"],
        fallback_used=pricing_result["fallback_used"],
        model_version="svd-collaborative-v1" if not pricing_result["fallback_used"] else "content-fallback-v1",
        insight=insight,
        comparables=comparables,
    )


# Interactive Web Dashboard for immediate browser demo
@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_DASHBOARD_UI


HTML_DASHBOARD_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Airbnb Dynamic Pricing Recommender | Member 2</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8fafc; }
    .airbnb-coral { color: #FF385C; }
    .bg-airbnb-coral { background-color: #FF385C; }
    .border-airbnb-coral { border-color: #FF385C; }
  </style>
</head>
<body class="text-slate-800 antialiased pb-16">

  <!-- Top Navbar -->
  <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#FF385C] to-[#E00B41] flex items-center justify-center text-white shadow-md shadow-rose-200">
          <i class="fa-solid fa-house-signal text-lg"></i>
        </div>
        <div>
          <h1 class="font-bold text-lg text-slate-900 leading-tight">Airbnb Dynamic Pricing</h1>
          <p class="text-xs text-slate-500 font-medium">Member 2: Pricing Engine + LLM Intelligence</p>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
          <span class="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
          Mock Data Mode Active
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto px-4 mt-6">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left Column: Controls & Target Listing (4 Cols) -->
      <div class="lg:col-span-5 space-y-5">
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h2 class="text-sm font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center">
            <i class="fa-solid fa-sliders mr-2 airbnb-coral"></i> Input Configuration
          </h2>

          <div class="space-y-4">
            <div>
              <label class="block text-xs font-bold text-slate-700 mb-1">Select Target Listing</label>
              <select id="listingSelect" class="w-full text-sm font-medium border border-slate-300 rounded-xl px-3 py-2.5 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-rose-500">
                <option value="101">101: 1BR in Williamsburg (Standard SVD)</option>
                <option value="102">102: Studio in East Village (High Demand)</option>
                <option value="103">103: 2BR Loft in Park Slope (Family)</option>
                <option value="104">104: Private Room in Midtown (Budget)</option>
                <option value="105">105: Bushwick 1BR (Cold-Start Fallback Test)</option>
              </select>
            </div>

            <div>
              <label class="block text-xs font-bold text-slate-700 mb-1">Pipeline Execution Mode</label>
              <div class="grid grid-cols-3 gap-2">
                <label class="cursor-pointer text-center py-2 px-1 rounded-lg border border-slate-200 text-xs font-medium text-slate-600 has-[:checked]:bg-rose-50 has-[:checked]:border-rose-500 has-[:checked]:text-rose-700">
                  <input type="radio" name="pipelineMode" value="pricing" class="sr-only">
                  <i class="fa-solid fa-calculator block mb-1"></i>
                  Pricing Only
                </label>
                <label class="cursor-pointer text-center py-2 px-1 rounded-lg border border-slate-200 text-xs font-medium text-slate-600 has-[:checked]:bg-rose-50 has-[:checked]:border-rose-500 has-[:checked]:text-rose-700">
                  <input type="radio" name="pipelineMode" value="insight" class="sr-only">
                  <i class="fa-solid fa-brain block mb-1"></i>
                  Insight Only
                </label>
                <label class="cursor-pointer text-center py-2 px-1 rounded-lg border border-slate-200 text-xs font-medium text-slate-600 has-[:checked]:bg-rose-50 has-[:checked]:border-rose-500 has-[:checked]:text-rose-700">
                  <input type="radio" name="pipelineMode" value="pricing_with_insight" class="sr-only" checked>
                  <i class="fa-solid fa-wand-magic-sparkles block mb-1"></i>
                  Price + LLM
                </label>
              </div>
            </div>

            <button id="runBtn" onclick="runRecommendation()" class="w-full bg-[#FF385C] hover:bg-[#E00B41] text-white font-bold py-3 px-4 rounded-xl shadow-md shadow-rose-200 transition-all flex items-center justify-center space-x-2">
              <i class="fa-solid fa-chart-line"></i>
              <span>Generate Recommendation</span>
            </button>
          </div>
        </div>

        <!-- Target Listing Card -->
        <div class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm" id="targetCard">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600" id="targetType">Entire home/apt</span>
            <span class="text-xs text-amber-600 font-semibold flex items-center">
              <i class="fa-solid fa-star mr-1"></i> <span id="targetRating">4.91</span> (<span id="targetReviews">84</span> reviews)
            </span>
          </div>
          <h3 class="font-bold text-base text-slate-900" id="targetName">Charming 1BR Apartment in Williamsburg</h3>
          <p class="text-xs text-slate-500 mt-1 flex items-center">
            <i class="fa-solid fa-location-dot mr-1.5 text-slate-400"></i> <span id="targetHood">Williamsburg, Brooklyn</span>
          </p>
          <div class="grid grid-cols-3 gap-2 mt-4 text-center">
            <div class="bg-slate-50 p-2 rounded-xl border border-slate-100">
              <span class="block text-xs text-slate-400 font-medium">Guests</span>
              <span class="font-bold text-slate-800 text-sm" id="targetGuests">2</span>
            </div>
            <div class="bg-slate-50 p-2 rounded-xl border border-slate-100">
              <span class="block text-xs text-slate-400 font-medium">Bedrooms</span>
              <span class="font-bold text-slate-800 text-sm" id="targetBeds">1</span>
            </div>
            <div class="bg-slate-50 p-2 rounded-xl border border-slate-100">
              <span class="block text-xs text-slate-400 font-medium">Bathrooms</span>
              <span class="font-bold text-slate-800 text-sm" id="targetBaths">1.0</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Recommendation Results & LLM (7 Cols) -->
      <div class="lg:col-span-7 space-y-5">

        <!-- Loading State -->
        <div id="loadingState" class="hidden bg-white rounded-2xl p-8 border border-slate-200 text-center shadow-sm">
          <div class="inline-block w-8 h-8 border-4 border-rose-500 border-t-transparent rounded-full animate-spin mb-3"></div>
          <p class="text-sm font-semibold text-slate-700">Calculating median peer rates & running LLM intelligence...</p>
        </div>

        <!-- Price Recommendation Hero Card -->
        <div id="pricingHeroCard" class="bg-gradient-to-br from-slate-900 via-slate-850 to-slate-900 text-white rounded-3xl p-6 shadow-xl relative overflow-hidden">
          <div class="absolute -right-8 -top-8 w-40 h-40 bg-rose-500/10 rounded-full blur-2xl"></div>
          
          <div class="flex items-start justify-between">
            <div>
              <span class="text-xs font-bold uppercase tracking-wider text-rose-400">Recommended Nightly Rate</span>
              <div class="flex items-baseline mt-1 space-x-2">
                <span class="text-5xl font-extrabold tracking-tight text-white" id="recPrice">$145</span>
                <span class="text-slate-400 text-sm font-medium">/ night</span>
              </div>
            </div>
            <span id="fallbackBadge" class="text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center">
              <i class="fa-solid fa-network-wired mr-1.5 text-xs"></i> SVD Collaborative Match
            </span>
          </div>

          <!-- Price Range Slider -->
          <div class="mt-6 pt-5 border-t border-slate-800">
            <div class="flex justify-between text-xs text-slate-400 font-medium mb-1.5">
              <span>Min Observed: <strong class="text-slate-200" id="pMin">$110</strong></span>
              <span>Median: <strong class="text-rose-400" id="pMed">$145</strong></span>
              <span>Max Observed: <strong class="text-slate-200" id="pMax">$190</strong></span>
            </div>
            <div class="w-full bg-slate-800 h-2.5 rounded-full relative overflow-hidden">
              <div class="bg-gradient-to-r from-emerald-400 via-rose-500 to-amber-400 h-full w-full rounded-full opacity-80"></div>
            </div>
            <p class="text-xs text-slate-400 mt-3 flex items-center justify-between">
              <span><i class="fa-solid fa-users mr-1"></i> Based on <strong class="text-white" id="compCount">8</strong> comparable listings</span>
              <span class="text-slate-500" id="modelVersionTag">Model: svd-collaborative-v1</span>
            </p>
          </div>
        </div>

        <!-- LLM AI Revenue Advisor Card -->
        <div id="llmInsightCard" class="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <div class="w-7 h-7 rounded-lg bg-indigo-100 text-indigo-600 flex items-center justify-center text-xs">
                <i class="fa-solid fa-brain"></i>
              </div>
              <h3 class="font-bold text-sm text-slate-900">AI Revenue Advisor (LLM)</h3>
            </div>
            <span class="text-xs font-bold px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200" id="llmTier">
              Market Median Tier
            </span>
          </div>

          <p class="text-sm font-semibold text-slate-800 bg-slate-50 p-3.5 rounded-xl border border-slate-100" id="llmHeadline">
            Recommended at $145/night based on 8 peer listings in Williamsburg.
          </p>

          <div>
            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Pricing Rationale</h4>
            <p class="text-xs text-slate-600 leading-relaxed" id="llmRationale">
              Loading rationale...
            </p>
          </div>

          <div>
            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Host Strategic Recommendations</h4>
            <ul class="space-y-2 text-xs text-slate-700" id="llmTips">
              <li class="flex items-start"><i class="fa-solid fa-circle-check text-rose-500 mt-0.5 mr-2"></i> Loading tips...</li>
            </ul>
          </div>

          <div class="pt-3 border-t border-slate-100 flex items-center text-xs text-slate-500">
            <i class="fa-solid fa-shield-check text-emerald-500 mr-2 text-sm"></i>
            <span id="llmConfidence">High confidence: Derived from collaborative review patterns.</span>
          </div>
        </div>

        <!-- Comparable Listings Table -->
        <div id="comparablesCard" class="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h3 class="font-bold text-sm text-slate-900 mb-3 flex items-center justify-between">
            <span><i class="fa-solid fa-list-check mr-2 airbnb-coral"></i> Top-K Comparable Peer Listings</span>
            <span class="text-xs text-slate-400 font-normal">Observed Market Prices</span>
          </h3>
          <div class="divide-y divide-slate-100" id="compsList">
            <!-- Injected dynamically -->
          </div>
        </div>

      </div>
    </div>
  </main>

  <script>
    async function runRecommendation() {
      const listingId = parseInt(document.getElementById('listingSelect').value);
      const mode = document.querySelector('input[name="pipelineMode"]:checked').value;
      const loading = document.getElementById('loadingState');
      const pricingCard = document.getElementById('pricingHeroCard');
      const llmCard = document.getElementById('llmInsightCard');
      const compsCard = document.getElementById('comparablesCard');

      loading.classList.remove('hidden');

      try {
        const res = await fetch('/api/v1/recommendations/price', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ listing_id: listingId, mode: mode })
        });
        const data = await res.json();

        // Update Target Card
        const t = data.target_listing;
        document.getElementById('targetName').innerText = t.name;
        document.getElementById('targetHood').innerText = t.neighborhood;
        document.getElementById('targetType').innerText = t.room_type || t.property_type;
        document.getElementById('targetRating').innerText = t.review_rating || 'N/A';
        document.getElementById('targetReviews').innerText = t.review_count || 0;
        document.getElementById('targetGuests').innerText = t.accommodates || 2;
        document.getElementById('targetBeds').innerText = t.bedrooms || 1;
        document.getElementById('targetBaths').innerText = t.bathrooms || 1.0;

        // Pricing UI
        if (mode === 'insight') {
          pricingCard.classList.add('hidden');
          compsCard.classList.add('hidden');
        } else {
          pricingCard.classList.remove('hidden');
          compsCard.classList.remove('hidden');
          document.getElementById('recPrice').innerText = '$' + Math.round(data.recommended_price);
          document.getElementById('pMin').innerText = '$' + Math.round(data.price_min);
          document.getElementById('pMed').innerText = '$' + Math.round(data.price_median);
          document.getElementById('pMax').innerText = '$' + Math.round(data.price_max);
          document.getElementById('compCount').innerText = data.comparable_count;
          document.getElementById('modelVersionTag').innerText = 'Model: ' + data.model_version;

          const fbBadge = document.getElementById('fallbackBadge');
          if (data.fallback_used) {
            fbBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation mr-1.5"></i> Content-Based Fallback';
            fbBadge.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center';
          } else {
            fbBadge.innerHTML = '<i class="fa-solid fa-network-wired mr-1.5"></i> SVD Collaborative Match';
            fbBadge.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center';
          }

          // Populate Comparables
          const compsList = document.getElementById('compsList');
          compsList.innerHTML = '';
          (data.comparables || []).forEach(c => {
            const row = document.createElement('div');
            row.className = 'py-2.5 flex items-center justify-between text-xs';
            row.innerHTML = `
              <div>
                <span class="font-bold text-slate-800">${c.name}</span>
                <span class="text-slate-400 ml-2">Similarity: ${(c.similarity_score * 100).toFixed(0)}%</span>
              </div>
              <span class="font-bold text-slate-900 bg-slate-100 px-2.5 py-1 rounded-lg">$${Math.round(c.price)} / night</span>
            `;
            compsList.appendChild(row);
          });
        }

        // LLM Insight UI
        if (mode === 'pricing') {
          llmCard.classList.add('hidden');
        } else if (data.insight) {
          llmCard.classList.remove('hidden');
          const ins = data.insight;
          document.getElementById('llmTier').innerText = ins.market_tier;
          document.getElementById('llmHeadline').innerText = ins.headline;
          document.getElementById('llmRationale').innerText = ins.pricing_rationale;
          document.getElementById('llmConfidence').innerText = ins.confidence_commentary;

          const tipsList = document.getElementById('llmTips');
          tipsList.innerHTML = '';
          (ins.actionable_tips || []).forEach(tip => {
            const li = document.createElement('li');
            li.className = 'flex items-start';
            li.innerHTML = '<i class="fa-solid fa-circle-check text-rose-500 mt-0.5 mr-2"></i>' + tip;
            tipsList.appendChild(li);
          });
        }
      } catch (err) {
        alert('Error fetching recommendation: ' + err.message);
      } finally {
        loading.classList.add('hidden');
      }
    }

    // Trigger on page load
    window.addEventListener('DOMContentLoaded', runRecommendation);
    document.getElementById('listingSelect').addEventListener('change', runRecommendation);
  </script>
</body>
</html>
"""
