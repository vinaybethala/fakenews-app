import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict


# Load model & tokenizer once (IMPORTANT)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()  # Set model to evaluation mode


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