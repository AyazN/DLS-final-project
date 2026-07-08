import re

def tokenize(text: str) -> list[str]:
    text = text.lower()

    # leave only digits and letters
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", " ", text)

    tokens = text.split()

    return tokens