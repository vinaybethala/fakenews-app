import requests
import os


# We'll load the API key lazily at runtime so dotenv has time to load
GOOGLE_FACTCHECK_API_KEY = None


def _get_api_key():
    global GOOGLE_FACTCHECK_API_KEY
    if not GOOGLE_FACTCHECK_API_KEY:
        GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    return GOOGLE_FACTCHECK_API_KEY


def verify_claim(claim: str) -> dict:
    """
    Sends a claim to Google Fact Check Tools API
    and returns verdict information.

    Returns:
        {
            "claim": str,
            "verdict": str,
            "publisher": str,
            "url": str
        }
    """

    key = _get_api_key()
    if not key:
        return {
            "claim": claim,
            "verdict": "API key missing",
            "publisher": None,
            "url": None
        }

    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": claim,
        "key": GOOGLE_FACTCHECK_API_KEY,
        "languageCode": "en"
    }

    try:
        response = requests.get(endpoint, params=params)
        data = response.json()

        # Check if API returned any claims
        if "claims" not in data or not data["claims"]:
            return {
                "claim": claim,
                "verdict": "No fact-check results found",
                "publisher": None,
                "url": None
            }

        first_claim = data["claims"][0]

        review = first_claim["claimReview"][0]

        return {
            "claim": claim,
            "verdict": review.get("textualRating", "No rating"),
            "publisher": review.get("publisher", {}).get("name", "Unknown"),
            "url": review.get("url", None)
        }

    except Exception as e:
        return {
            "claim": claim,
            "verdict": f"Error: {str(e)}",
            "publisher": None,
            "url": None
        }   