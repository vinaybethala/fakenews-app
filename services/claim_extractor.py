import spacy
from typing import List

nlp = spacy.load("en_core_web_sm")

def is_factual_sentence(sent) -> bool:
    """
    Determine whether a sentence looks like a factual claim.
    """

    # Condition 1: Contains numbers
    contains_number = any(token.like_num for token in sent)

    # Condition 2: Contains named entities
    contains_entity = any(ent.label_ in [
        "PERSON", "ORG", "GPE", "DATE", "MONEY", "PERCENT"
    ] for ent in sent.ents)

    # Condition 3: Contains strong action verbs
    strong_verbs = {"increase", "decrease", "announce", "report",
                    "claim", "state", "declare", "confirm",
                    "win", "lose", "launch", "ban"}

    contains_strong_verb = any(
        token.lemma_.lower() in strong_verbs and token.pos_ == "VERB"
        for token in sent
    )

    # Final decision logic
    return contains_number or contains_entity or contains_strong_verb


def extract_claims(text: str) -> List[str]:
    """
    Extract factual claims from news article.
    """

    if not text:
        return []

    doc = nlp(text)

    claims = []

    for sent in doc.sents:
        if is_factual_sentence(sent):
            claims.append(sent.text.strip())

    # Remove duplicates
    unique_claims = list(dict.fromkeys(claims))

    return unique_claims