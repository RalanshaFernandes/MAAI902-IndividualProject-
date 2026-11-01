"""Extract plain text from the downloaded PDFs with PyMuPDF.

For each PDF we record simple quality signals used later in pre-processing:
  - chars_per_page: very low values mean a scanned image with no text layer
  - arabic_ratio:   share of letters that are Arabic, used to tag the language

Output: data/processed/text_en/<SYMBOL>/<attachment_id>.txt
        data/processed/documents.csv (one row per PDF with metadata + quality signals)
"""
import re
import sys

import pymupdf as fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

from config import PDF_DIR, PROCESSED, RAW, TEXT_DIR

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
SCANNED_THRESHOLD = 200  # fewer extracted characters per page than this = likely scanned


def language_of(text: str) -> tuple[str, float]:
    ar, en = len(ARABIC.findall(text)), len(LATIN.findall(text))
    ratio = ar / (ar + en) if ar + en else 0.0
    lang = "ar" if ratio > 0.7 else "en" if ratio < 0.3 else "mixed"
    return lang, round(ratio, 3)


def main():
    att = pd.read_csv(RAW / "attachments.csv", parse_dates=["published"])
    # The API's own language tag says "en" for almost everything, so we replace
    # it with a language detected from the extracted text.
    att = att.drop(columns=["language"])
    # Incremental: documents already in documents.csv keep their text (and OCR),
    # so a data refresh only extracts newly downloaded PDFs. --all redoes everything.
    old = pd.DataFrame(columns=["attachment_id"])
    if (PROCESSED / "documents.csv").exists() and "--all" not in sys.argv:
        old = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["published"])
    rows = []
    pdfs = [p for p in sorted(PDF_DIR.glob("*/*.pdf")) if p.stem not in set(old["attachment_id"])]

    for pdf in tqdm(pdfs, desc="Extracting"):
        out = TEXT_DIR / pdf.parent.name / f"{pdf.stem}.txt"
        try:
            with fitz.open(pdf) as doc:
                n_pages = doc.page_count
                text = "\n\f".join(page.get_text("text") for page in doc)  # \f marks page breaks
        except Exception as e:
            rows.append({"attachment_id": pdf.stem, "error": str(e)[:200]})
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        lang, ratio = language_of(text)
        chars_per_page = len(text.strip()) / max(n_pages, 1)
        rows.append({
            "attachment_id": pdf.stem,
            "n_pages": n_pages,
            "n_chars": len(text),
            "chars_per_page": round(chars_per_page),
            "arabic_ratio": ratio,
            "language": lang,
            "scanned": chars_per_page < SCANNED_THRESHOLD,
            "text_path": str(out.relative_to(TEXT_DIR)),
        })

    new = att.merge(pd.DataFrame(rows), on="attachment_id", how="inner") if rows else att.iloc[:0]
    docs = pd.concat([old, new], ignore_index=True) if len(old) else new
    print(f"{len(new)} new documents extracted")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    docs.to_csv(PROCESSED / "documents.csv", index=False, encoding="utf-8-sig")

    print(f"\n{len(docs)} documents extracted")
    print("Language:\n", docs["language"].value_counts().to_string())
    print(f"Likely scanned (no text layer): {docs['scanned'].sum()}")
    print(f"Median pages: {docs['n_pages'].median():.0f}, median chars: {docs['n_chars'].median():.0f}")


if __name__ == "__main__":
    main()
