import re


def tokenize(text: str) -> list[str]:
    text = text.lower()

    # leave only digits and letters
    text = re.sub(r"[^a-zA-Z\u0430-\u044f\u0410-\u042f0-9\s]", " ", text)

    tokens = text.split()

    return tokens
