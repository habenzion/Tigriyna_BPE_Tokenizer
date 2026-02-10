import unicodedata
import regex as re

WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text