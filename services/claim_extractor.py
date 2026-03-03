import re
from typing import List

try:
    import spacy
except Exception:
    spacy = None


# Try to load spaCy model once; if unavailable we'll use a safe text-only fallback
nlp = None
if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None


def _is_factual_spacy_sent(sent) -> bool:
    # Ignore very short sentences
    if len(sent.text.strip()) < 6:
        return False

    # Must contain at least one verb and not be a question
    contains_verb = any(token.pos_ == "VERB" for token in sent)
    is_question = sent.text.strip().endswith("?")
    return contains_verb and not is_question


def _is_factual_text_sent(sent_text: str) -> bool:
    # Basic fallback heuristics for claim-like sentences
    s = sent_text.strip()
    if len(s) < 6:
        return False
    if s.endswith('?'):
        return False

    # If contains a number it's likely factual
    if re.search(r"\d", s):
        return True

    # Strong verb keywords (same intent as spaCy-based detector)
    strong_verbs = {"increase", "decrease", "announce", "report",
                    "claim", "state", "declare", "confirm",
                    "win", "lose", "launch", "ban"}
    words = [w.strip('.,') for w in s.lower().split()]
    if any(w in strong_verbs for w in words):
        return True

    # If sentence contains multiple Titlecase words (possible proper nouns)
    titlecase_count = sum(1 for w in s.split() if w[0].isupper())
    if titlecase_count >= 2:
        return True

    return False


def extract_claims(text: str) -> List[str]:
    """Extract factual claims from a news article.

    Uses spaCy sentence segmentation and heuristics when available. Falls
    back to a regex-based sentence splitter and lightweight heuristics when
    spaCy or the model cannot be loaded.
    """

    if not text or not text.strip():
        return []

    claims: List[str] = []

    if nlp is not None:
        doc = nlp(text)
        for sent in doc.sents:
            if _is_factual_spacy_sent(sent):
                claims.append(sent.text.strip())
    else:
        # Fallback: split on sentence boundaries and apply simple heuristics
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for s in sentences:
            if _is_factual_text_sent(s):
                claims.append(s.strip())

    # Remove duplicates while preserving order
    unique_claims = list(dict.fromkeys([c for c in claims if c]))
    return unique_claims
