from dotenv import load_dotenv, find_dotenv

# ensure .env variables are available before any service modules are imported
# use find_dotenv to locate file even if cwd changes
load_dotenv(find_dotenv())

import os
from flask import Flask, render_template, request, jsonify

# debug log on startup to check API key presence
_api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
print("FactCheck API key loaded:", "YES" if _api_key else "NO")

from services.preprocess import clean_text
from services.predictor import predict_news
from services.claim_extractor import extract_claims
from services.factcheck_api import verify_claim

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        original_text = data["text"]

        # 1️⃣ Clean text
        cleaned_text = clean_text(original_text)

        # 2️⃣ Prediction
        prediction_result = predict_news(cleaned_text)

        # 3️⃣ Extract claims
        claims = extract_claims(original_text)

        # 4️⃣ Fact-check claims (limit to prevent API overload)
        fact_results = []
        for claim in claims[:5]:
            fact_results.append(verify_claim(claim))

        return jsonify({
            "prediction": prediction_result,
            "claims": claims,
            "fact_checks": fact_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)