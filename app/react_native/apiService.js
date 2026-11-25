/**
 * API Service for communicating with the FastAPI Pricing & LLM Backend.
 * Member 2 Responsibility: Client integration contract.
 */

const API_BASE_URL = 'http://10.0.2.2:8000/api/v1'; // Android emulator localhost or actual backend IP

export const fetchPriceRecommendation = async (listingId = 101, mode = 'pricing_with_insight') => {
  try {
    const response = await fetch(`${API_BASE_URL}/recommendations/price`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        listing_id: listingId,
        mode: mode,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching dynamic price recommendation:', error);
    throw error;
  }
};

export const fetchSampleListings = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/listings`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching sample listings:', error);
    return [];
  }
};
