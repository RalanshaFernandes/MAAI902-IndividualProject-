"""Text cleaning and sentence segmentation for disclosure documents.

PDF and OCR text is noisy: lines are broken mid-sentence, tables become
streams of numbers, letterheads repeat addresses and phone numbers, and the
English OCR of bilingual letters turns the Arabic half into gibberish. These
functions turn a raw document into a list of clean English sentences that a
sentence-level model such as FinBERT can score.
"""
import re

# Boilerplate patterns removed before segmentation.
_PATTERNS = [
    re.compile(r"page\s+\d+\s+(of|/)\s+\d+", re.I),     # "Page 3 of 42"
    re.compile(r"\S+@\S+\.\S+"),                         # e-mail addresses
    re.compile(r"(https?://|www\.)\S+", re.I),           # URLs
    re.compile(r"\+?\d[\d\s\-()]{7,}\d"),                # phone / fax numbers
    re.compile(r"p\.?\s?o\.?\s?box\s*\d+", re.I),        # P.O. Box 9440
]
_ARABIC = re.compile(r"[؀-ۿ]+")
# A "word-like" token: ordinary words, acronyms, numbers, money and percentages.
_WORDLIKE = re.compile(r"^([A-Za-z][a-z'\-]+|[A-Z]{2,}|[A-Za-z]|AED|\d[\d,.]*%?|[(\[]?\d[\d,.]*[)\]]?)[.,;:)]?$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")

MIN_WORDS, MAX_WORDS = 6, 80
MIN_WORDLIKE = 0.75   # share of tokens that must look like real words
MIN_ALPHA = 0.5       # share of tokens containing letters (drops number-only table rows)


def clean(text: str) -> str:
    """Remove boilerplate and re-join lines that PDF extraction broke apart."""
    text = text.replace("\f", "\n\n")
    text = _ARABIC.sub(" ", text)              # Arabic is handled separately (future work)
    for p in _PATTERNS:
        text = p.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    text = re.sub(r"[•▪●◦•]", " ", text)             # bullet glyphs
    text = re.sub(r"[ \t]+", " ", text)
    # Single newlines are line wraps; blank lines are paragraph breaks.
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n".join(p for p in paragraphs if p)


def is_good_sentence(sentence: str) -> bool:
    """Keep only well-formed prose; reject tables, headers and OCR gibberish."""
    tokens = sentence.split()
    if not MIN_WORDS <= len(tokens) <= MAX_WORDS:
        return False
    wordlike = sum(bool(_WORDLIKE.match(t)) for t in tokens) / len(tokens)
    alpha = sum(any(c.isalpha() for c in t) for t in tokens) / len(tokens)
    return wordlike >= MIN_WORDLIKE and alpha >= MIN_ALPHA


def sentences(text: str, max_sentences: int | None = None) -> list[str]:
    """Clean a document and return its good sentences in reading order."""
    out = []
    for paragraph in clean(text).split("\n"):
        for s in _SENT_SPLIT.split(paragraph):
            s = s.strip()
            if is_good_sentence(s):
                out.append(s)
                if max_sentences and len(out) >= max_sentences:
                    return out
    return out
