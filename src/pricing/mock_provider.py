"""
Mock Data Provider for Airbnb Dynamic Pricing Recommender.
Simulates Member 1's SVD collaborative filtering & content-based outputs
to allow Member 2 to build and test the pricing, LLM, API, and UI layers independently.
"""

from typing import Dict, Any, List

# Catalog of sample target listings available for demonstration
SAMPLE_TARGET_LISTINGS: Dict[int, Dict[str, Any]] = {
    101: {
        "id": 101,
        "name": "Charming 1BR Apartment in Williamsburg",
        "neighborhood": "Williamsburg, Brooklyn",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 2,
        "bedrooms": 1,
        "bathrooms": 1.0,
        "amenities": ["Wifi", "Air conditioning", "Kitchen", "Dedicated workspace", "Self check-in"],
        "historical_price": 140.0,
        "review_rating": 4.91,
        "review_count": 84,
    },
    102: {
        "id": 102,
        "name": "Sunlit Studio in Heart of East Village",
        "neighborhood": "East Village, Manhattan",
        "room_type": "Entire home/apt",
        "property_type": "Entire condominium",
        "accommodates": 2,
        "bedrooms": 0,
        "bathrooms": 1.0,
        "amenities": ["Wifi", "Heating", "Hair dryer", "Coffee maker"],
        "historical_price": 175.0,
        "review_rating": 4.85,
        "review_count": 122,
    },
    103: {
        "id": 103,
        "name": "Spacious 2BR Family Loft near Prospect Park",
        "neighborhood": "Park Slope, Brooklyn",
        "room_type": "Entire home/apt",
        "property_type": "Entire loft",
        "accommodates": 5,
        "bedrooms": 2,
        "bathrooms": 2.0,
        "amenities": ["Wifi", "Washer", "Dryer", "Kitchen", "Patio", "Crib"],
        "historical_price": 260.0,
        "review_rating": 4.96,
        "review_count": 53,
    },
    104: {
        "id": 104,
        "name": "Cozy Private Room in Midtown Manhattan",
        "neighborhood": "Midtown, Manhattan",
        "room_type": "Private room",
        "property_type": "Private room in apartment",
        "accommodates": 1,
        "bedrooms": 1,
        "bathrooms": 1.0,
        "amenities": ["Wifi", "Elevator", "Air conditioning"],
        "historical_price": 95.0,
        "review_rating": 4.72,
        "review_count": 39,
    },
    105: {
        "id": 105,
        "name": "Brand New Modern 1BR (Cold-Start Demonstration)",
        "neighborhood": "Bushwick, Brooklyn",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 2,
        "bedrooms": 1,
        "bathrooms": 1.0,
        "amenities": ["Wifi", "Smart TV", "Modern Kitchen"],
        "historical_price": None,
        "review_rating": None,
        "review_count": 0,  # Cold-start case: 0 reviews!
    },
}

# Mock Top-K comparables simulating Member 1 SVD / Content output
MOCK_COMPARABLES_STORE: Dict[int, List[Dict[str, Any]]] = {
    101: [
        {"id": 2001, "name": "Cozy Williamsburg 1BR Walk-up", "neighborhood": "Williamsburg", "price": 145.0, "similarity_score": 0.94, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.88},
        {"id": 2002, "name": "Modern Williamsburg Apt w/ Balcony", "neighborhood": "Williamsburg", "price": 160.0, "similarity_score": 0.91, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.92},
        {"id": 2003, "name": "Rustic 1BR Oasis near Bedford L Train", "neighborhood": "Williamsburg", "price": 140.0, "similarity_score": 0.89, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.85},
        {"id": 2004, "name": "Sunny Northside Williamsburg Flat", "neighborhood": "Williamsburg", "price": 150.0, "similarity_score": 0.87, "accommodates": 3, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.90},
        {"id": 2005, "name": "Minimalist Brooklyn 1BR Retreat", "neighborhood": "Greenpoint", "price": 135.0, "similarity_score": 0.84, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.79},
        {"id": 2006, "name": "Vintage Williamsburg Loft Style 1BR", "neighborhood": "Williamsburg", "price": 175.0, "similarity_score": 0.82, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.95},
        {"id": 2007, "name": "Quiet East River View Apartment", "neighborhood": "Williamsburg", "price": 180.0, "similarity_score": 0.80, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.89},
        {"id": 2008, "name": "Budget-Friendly 1BR near Graham Ave", "neighborhood": "Williamsburg", "price": 120.0, "similarity_score": 0.77, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.70},
    ],
    102: [
        {"id": 2101, "name": "Bright East Village Studio on St Marks", "neighborhood": "East Village", "price": 180.0, "similarity_score": 0.95, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.89},
        {"id": 2102, "name": "Chic Studio Flat in Alphabet City", "neighborhood": "East Village", "price": 165.0, "similarity_score": 0.92, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.82},
        {"id": 2103, "name": "Historic East Village Artist Studio", "neighborhood": "East Village", "price": 195.0, "similarity_score": 0.89, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.93},
        {"id": 2104, "name": "Cozy Lower East Side Studio", "neighborhood": "Lower East Side", "price": 170.0, "similarity_score": 0.86, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.80},
        {"id": 2105, "name": "Quiet Garden View Manhattan Studio", "neighborhood": "East Village", "price": 175.0, "similarity_score": 0.84, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.87},
        {"id": 2106, "name": "Modern Bowery Studio w/ Gym", "neighborhood": "Lower East Side", "price": 210.0, "similarity_score": 0.81, "accommodates": 2, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.91},
        {"id": 2107, "name": "Compact St Marks Studio", "neighborhood": "East Village", "price": 155.0, "similarity_score": 0.79, "accommodates": 1, "bedrooms": 0, "bathrooms": 1.0, "rating": 4.68},
    ],
    103: [
        {"id": 2201, "name": "Brownstone 2BR Flat on 8th Ave", "neighborhood": "Park Slope", "price": 270.0, "similarity_score": 0.93, "accommodates": 4, "bedrooms": 2, "bathrooms": 2.0, "rating": 4.97},
        {"id": 2202, "name": "Prospect Park South 2BR Sanctuary", "neighborhood": "Park Slope", "price": 255.0, "similarity_score": 0.90, "accommodates": 5, "bedrooms": 2, "bathrooms": 1.5, "rating": 4.92},
        {"id": 2203, "name": "Spacious Park Slope Garden Duplex", "neighborhood": "Park Slope", "price": 290.0, "similarity_score": 0.88, "accommodates": 6, "bedrooms": 2, "bathrooms": 2.0, "rating": 4.98},
        {"id": 2204, "name": "Sun-drenched 2BR near 7th Ave Subway", "neighborhood": "Park Slope", "price": 240.0, "similarity_score": 0.85, "accommodates": 4, "bedrooms": 2, "bathrooms": 1.0, "rating": 4.88},
        {"id": 2205, "name": "Family Friendly Brownstone Loft", "neighborhood": "Prospect Heights", "price": 265.0, "similarity_score": 0.83, "accommodates": 5, "bedrooms": 2, "bathrooms": 2.0, "rating": 4.94},
        {"id": 2206, "name": "Charming Gowanus 2BR Artist Home", "neighborhood": "Gowanus", "price": 225.0, "similarity_score": 0.79, "accommodates": 4, "bedrooms": 2, "bathrooms": 1.0, "rating": 4.81},
    ],
    104: [
        {"id": 2301, "name": "Private Room near Times Square", "neighborhood": "Midtown", "price": 95.0, "similarity_score": 0.94, "accommodates": 1, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.75},
        {"id": 2302, "name": "Hell's Kitchen Cozy Room", "neighborhood": "Hell's Kitchen", "price": 105.0, "similarity_score": 0.91, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.81},
        {"id": 2303, "name": "Quiet Midtown East Bedroom", "neighborhood": "Midtown East", "price": 90.0, "similarity_score": 0.88, "accommodates": 1, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.70},
        {"id": 2304, "name": "Convenient Penn Station Room", "neighborhood": "Midtown", "price": 110.0, "similarity_score": 0.85, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.80},
        {"id": 2305, "name": "Clean Budget Bedroom near Grand Central", "neighborhood": "Murray Hill", "price": 85.0, "similarity_score": 0.82, "accommodates": 1, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.65},
    ],
    105: [
        # Cold start demonstration: Only 3 content-based matches found!
        {"id": 2401, "name": "Bushwick Loft 1BR near Jefferson L", "neighborhood": "Bushwick", "price": 125.0, "similarity_score": 0.88, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.80},
        {"id": 2402, "name": "Modern Industrial Bushwick Unit", "neighborhood": "Bushwick", "price": 135.0, "similarity_score": 0.84, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.75},
        {"id": 2403, "name": "Cozy Brooklyn 1BR Artist Pad", "neighborhood": "Bushwick", "price": 115.0, "similarity_score": 0.81, "accommodates": 2, "bedrooms": 1, "bathrooms": 1.0, "rating": 4.70},
    ],
}


def get_target_listing(listing_id: int) -> Dict[str, Any]:
    """Retrieve sample listing metadata by ID."""
    return SAMPLE_TARGET_LISTINGS.get(listing_id, SAMPLE_TARGET_LISTINGS[101])


def get_mock_comparables(listing_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve Top-K comparable listings for a target listing.
    Returns realistic comparable listings simulating Member 1's model output.
    """
    return MOCK_COMPARABLES_STORE.get(listing_id, MOCK_COMPARABLES_STORE[101])
