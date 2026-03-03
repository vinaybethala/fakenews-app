import requests
import os

# no caching; read from environment each time so updates are picked up and
# avoid freezing None if the variable wasn't set when module imported


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

    key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
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
        "key": key,
        "languageCode": "en"
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "claims" not in data or not data["claims"]:
            return {
                "claim": claim,
                "verdict": "No fact-check results found",
                "publisher": None,
                "url": None
            }

        first_claim = data["claims"][0]
        if "claimReview" not in first_claim or not first_claim["claimReview"]:
            return {
                "claim": claim,
                "verdict": "No review available",
                "publisher": None,
                "url": None
            }

        review = first_claim["claimReview"][0]

        return {
            "claim": claim,
            "verdict": review.get("textualRating", "No rating"),
            "publisher": review.get("publisher", {}).get("name", "Unknown"),
            "url": review.get("url", None)
        }

    except requests.exceptions.Timeout:
        return {
            "claim": claim,
            "verdict": "Request timeout",
            "publisher": None,
            "url": None
        }
    except requests.exceptions.HTTPError as e:
        return {
            "claim": claim,
            "verdict": f"API error: {e.response.status_code}",
            "publisher": None,
            "url": None
        }
    except Exception as e:
        return {
            "claim": claim,
            "verdict": f"Error: {type(e).__name__}",
            "publisher": None,
            "url": None
        }   