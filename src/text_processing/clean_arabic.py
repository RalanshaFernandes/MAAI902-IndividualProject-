"""Arabic text handling: normalisation and sentence segmentation.

PDF extraction of Arabic often returns Unicode "presentation forms" (separate
code points for the initial / medial / final shapes of each letter), plus
diacritics (harakat) and tatweel (the stretching character U+0640). Models are
trained on plain letters, so we normalise:
  * NFKC: presentation forms -> standard Arabic letters
  * remove diacritics (U+064B-U+0652, U+0670) and tatweel
  * unify alef variants (أ إ آ -> ا), a common, low-risk normalisation
We deliberately do NOT merge ى/ي or ة/ه, which changes meaning more often.

Two PDF-specific errors are then repaired with a lexicon (see repair()):
  * reversed lam-alef: many DFM PDFs map the لا ligature glyph to "ال", so
    "خلال" (during) is extracted as "خالل" and "الإمارات" as "االمارات".
  * split words: some PDFs insert spaces inside words ("دب ي" for "دبي").
"""
import re
import unicodedata
from functools import lru_cache
from itertools import combinations

DIACRITICS = re.compile(r"[ً-ْٰ]")
TATWEEL = "ـ"
ALEF = re.compile(r"[آأإ]")
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
# Sentence ends: Latin/Arabic full stop, Arabic question mark and semicolon, line breaks between paragraphs.
SPLIT = re.compile(r"(?<=[.!؟?؛])\s+|\n\s*\n")

MIN_WORDS, MAX_WORDS, MIN_ARABIC_SHARE = 4, 80, 0.7


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = ALEF.sub("ا", text)
    text = re.sub(r"\S+@\S+|(https?://|www\.)\S+|\+?\d[\d\s\-()]{7,}\d", " ", text)   # emails, URLs, phones
    return text


TOKEN = re.compile(r"[ء-ي]+")


@lru_cache(maxsize=1)
def lexicon() -> frozenset:
    """Known Arabic words: CAMeLBERT's vocabulary + correctly encoded Arabic text we hold
    (DFM Arabic headlines from the API, Mubasher Arabic headlines and leads)."""
    from pathlib import Path

    import pandas as pd
    from transformers import AutoTokenizer

    from config import PROCESSED, RAW
    words = set()
    local = Path.home() / "models" / "bert-base-arabic-camelbert-msa-sentiment"
    tok = AutoTokenizer.from_pretrained(str(local) if local.exists() else "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment")
    words |= {normalise(w) for w in tok.get_vocab() if not w.startswith("##") and TOKEN.fullmatch(w)}
    texts = list(pd.read_csv(RAW / "disclosures.csv")["headline_ar"].dropna())
    mub = PROCESSED / "mubasher.csv"
    if mub.exists():
        m = pd.read_csv(mub)
        m = m[m["lang"] == "ar"]
        texts += list(m["headline"].dropna()) + list(m["lead"].dropna())
    for t in texts:
        words |= set(TOKEN.findall(normalise(t)))
    return frozenset(w for w in words if len(w) >= 2)


def _fix_lam_alef(word: str, lex: frozenset) -> str:
    """Swap reversed "ال" back to "لا" where that turns a non-word into a known word."""
    if word in lex or "ال" not in word:
        return word
    spots = [m.start() for m in re.finditer("(?=ال)", word)]
    for k in range(1, min(len(spots), 3) + 1):
        for chosen in combinations(spots, k):
            w = list(word)
            for i in chosen:
                w[i], w[i + 1] = "ل", "ا"
            cand = "".join(w)
            if cand in lex:
                return cand
    return word


def repair(text: str) -> str:
    """Fix reversed lam-alef and words split by stray spaces, using the lexicon."""
    lex = lexicon()
    text = TOKEN.sub(lambda m: _fix_lam_alef(m.group(0), lex), text)
    # Merge neighbouring fragments when they are not both words but their join is.
    def merge(m):
        a, b = m.group(1), m.group(2)
        joined = _fix_lam_alef(a + b, lex)
        return joined if joined in lex and not (a in lex and b in lex) else m.group(0)
    for _ in range(2):   # two passes catch words split into three pieces
        text = re.sub(r"([ء-ي]+) ([ء-ي]+)", merge, text)
    return text


def oov_rate(text: str) -> float:
    """Share of Arabic word tokens that are not in the lexicon (lower = cleaner text)."""
    toks = TOKEN.findall(text)
    return sum(t not in lexicon() for t in toks) / len(toks) if toks else 0.0


def arabic_share(text: str) -> float:
    a, l = len(ARABIC.findall(text)), len(LATIN.findall(text))
    return a / (a + l) if a + l else 0.0


def sentences(text: str, max_sentences: int | None = None) -> list[str]:
    """Normalised Arabic sentences that are long enough and mostly Arabic script."""
    text = repair(normalise(text)).replace("\f", "\n\n")
    out = []
    for para in SPLIT.split(text):
        s = re.sub(r"\s+", " ", para).strip()
        n = len(s.split())
        if MIN_WORDS <= n <= MAX_WORDS and arabic_share(s) >= MIN_ARABIC_SHARE:
            out.append(s)
            if max_sentences and len(out) >= max_sentences:
                break
    return out
