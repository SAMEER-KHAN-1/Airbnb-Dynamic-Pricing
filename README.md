# Airbnb Dynamic Pricing Recommender

End-to-end Machine Learning Dynamic Pricing Recommender system for short-term rental listings using Inside Airbnb marketplace data.

---

## Architecture Overview

```
                                      ┌─────────────────────────────────────┐
                                      │   Member 1: Recommender Core        │
                                      │   - User-Item Matrix & SVD          │
                                      │   - Content-Based Matcher           │
                                      │   - Top-K Comparable Listing IDs    │
                                      └──────────────────┬──────────────────┘
                                                         │
                                               [Top-K Comparables Handoff]
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Member 2: Pricing Engine + LLM Intelligence + Application Layer                                        │
│                                                                                                        │
│   ┌──────────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────────┐  │
│   │ 1. Pricing Engine        │      │ 2. LLM Intelligence      │      │ 3. Application UI           │  │
│   │ - Clean & filter prices  │ ───► │ - Context Builder        │ ───► │ - Web Dashboard             │  │
│   │ - IQR outlier rejection  │      │ - Persona & Strategy     │      │ - React Native Mobile App   │  │
│   │ - Median price & bounds  │      │ - Pydantic Validation    │      │ - Real-time Mode Switching  │  │
│   │ - Cold-start fallback    │      │ - Host Actionable Advice │      │   (pricing, insight, both)  │  │
│   └──────────────────────────┘      └──────────────────────────┘      └─────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## What is Implemented (Member 2 Scope)

1. **Pricing Aggregation Engine (`src/pricing/pricing_engine.py`)**:
   - Cleans observed prices and filters corrupt / extreme price spikes using Interquartile Range (IQR).
   - Computes market-reference price statistics (Min, 25th percentile, Median, 75th percentile, Max).
   - Implements cold-start fallback detection: if fewer than 5 peer matches are available, automatically activates content-based fallback.
2. **Mock Comparable Provider (`src/pricing/mock_provider.py`)**:
   - Supplies realistic candidate listings and observed peer rates to allow parallel development without waiting for model training.
3. **LLM Intelligence Pipeline (`src/llm/llm_pipeline.py`)**:
   - Converts raw mathematical results into compact structured JSON.
   - Enforces strict Pydantic schemas (`PricingInsightSchema`) for reliable parsing.
   - Generates executive headline, competitive market positioning, pricing rationale, host optimization tips, and confidence commentary.
4. **FastAPI Backend Service (`src/api/main.py`)**:
   - Validated REST API endpoints supporting 3 execution modes:
     - `pricing`: Fast numerical recommendation only.
     - `insight`: LLM revenue analysis only.
     - `pricing_with_insight`: Full end-to-end price computation + LLM explanation.
5. **Interactive UI**:
   - **Live Web Dashboard (`http://127.0.0.1:8000`)**: Interactive Airbnb-styled dashboard for instant desktop presentation.
   - **React Native Mobile App (`app/react_native/PricingScreen.jsx`)**: Cross-platform mobile UI component with API client.
6. **Automated Test Suite (`tests/test_pricing.py`)**:
   - 5 comprehensive unit tests verifying outlier removal, median aggregation, cold-start fallback flags, and LLM schema compliance.

---

## Quick Start (Demo for Presentation)

### Step 1: Run the Interactive Demo Dashboard
Open PowerShell or Terminal inside this directory, install the project dependencies, and start the demo:

```powershell
python -m pip install -r requirements.txt
python run_demo.py
```

This will automatically launch the FastAPI server and open your default browser at `http://127.0.0.1:8000`.

### Step 2: Running Automated Tests
To demonstrate verification to evaluators:

```bash
python -m pytest tests/test_pricing.py -v
```

---

## Deploy for a Public Demo

The included `render.yaml` configures the FastAPI dashboard for Render.

1. Make sure this project is available in a GitHub repository under your account. I have not performed any Git or GitHub operations.
2. In Render, choose **New + → Blueprint**, connect your repository, and select this project.
3. Render will read `render.yaml`, install `requirements.txt`, and start the web service.
4. When deployment finishes, open the `onrender.com` URL Render provides and share it with your teacher.

The free service may pause after inactivity, so its first visit can take a little while to start.

## API Endpoints Reference

- `GET /` — Interactive Web Dashboard
- `POST /api/v1/recommendations/price` — Main recommendation endpoint (Pricing, Insight, or Both)
- `GET /api/v1/listings` — List sample target listings
- `GET /api/v1/listings/{id}` — Fetch specific listing details
- `GET /api/v1/health` — Health check and model readiness status
