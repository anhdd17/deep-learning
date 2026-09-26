import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()
