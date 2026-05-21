import re


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)