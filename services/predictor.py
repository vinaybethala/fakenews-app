
from typing import Dict

# try to import heavy dependencies; if they're missing we fall back
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore


# Load model & tokenizer once (IMPORTANT)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# initialize to None and then attempt to load
tokenizer = None
model = None

if torch is not None and AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()  # Set to evaluation mode
    except Exception:
        # loading failure, leave model/tokenizer as None
        tokenizer = None
        model = None


def predict_news(text: str) -> Dict[str, float]:
    """
    Predict whether news text is FAKE or REAL.

    Returns:
        {
            "label": "FAKE" or "REAL",
            "confidence": float (percentage)
        }
    """

    if not text or not text.strip():
        return {
            "label": "UNKNOWN",
            "confidence": 0.0
        }

    # if the heavy dependencies failed to import or model didn't load, return safe default
    if torch is None or tokenizer is None or model is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0
        }

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Disable gradient calculation (faster inference)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

    confidence = confidence.item() * 100
    predicted_class = predicted_class.item()

    # Map model labels to Fake/Real
    # SST-2: 0 = Negative, 1 = Positive
    if predicted_class == 1:
        label = "REAL"
    else:
        label = "FAKE"

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }