"""
Demo Runner for Airbnb Dynamic Pricing Recommender (Member 2).
Launches the FastAPI server and automatically opens the interactive Web UI in your default browser.
"""

import uvicorn
import webbrowser
import threading
import time
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[DEMO] Opening interactive UI in browser at: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    print("=" * 60)
    print("Airbnb Dynamic Pricing Recommender - Member 2 Demo")
    print("FastAPI Backend + ML Pricing Aggregator + LLM Revenue Advisor")
    print("Running in MOCK DATA mode for parallel demonstration")
    print("=" * 60)

    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
