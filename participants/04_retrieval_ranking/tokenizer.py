import re

def tokenize(text: str) -> list[str]:
    text = text.lower()

    # Keep Latin and Cyrillic letters, digits, and whitespace.
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", " ", text)

    tokens = text.split()

    return tokens