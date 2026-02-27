import re


def clean_text(text: str) -> str:
    """
    Clean raw news text before sending to NLP pipeline.

    Steps:
    1. Remove URLs
    2. Convert text to lowercase
    3. Remove unwanted special characters
    4. Normalize multiple spaces
    5. Strip leading/trailing spaces

    Returns:
        Cleaned string
    """

    if not text:
        return ""

    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 2. Lowercase normalization
    text = text.lower()

    # 3. Remove special characters (keep letters, numbers, spaces, punctuation needed)
    text = re.sub(r'[^\w\s\.,!?]', '', text)

    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 5. Trim spaces
    text = text.strip()

    return text
sample = "   Helllo  word  "
# sample = "BREAKING NEWS!!! Visit https://news.com NOW 😡😡More updates soon."

print(clean_text(sample))