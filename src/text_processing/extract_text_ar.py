"""Extract text from the Arabic disclosure PDFs (with Arabic OCR for scans).

Input:  data/raw/pdfs/ar/<SYMBOL>/<attachment_id>.pdf (from download_pdfs.py --lang ar)
Output: data/processed/text_ar/<SYMBOL>/<attachment_id>.txt
        data/processed/documents_ar.csv (metadata + quality signals + English twin id)
"""
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pymupdf
import pytesseract
from PIL import Image
from tqdm import tqdm

from text_processing.clean_arabic import arabic_share
from config import PDF_DIR, PROCESSED, RAW, TEXT_AR_DIR

TEXT_AR = TEXT_AR_DIR
SCANNED_THRESHOLD = 200
OCR_MAX_PAGES, DPI, MAX_PIXELS = 5, 300, 40_000_000
# Tesseract with Arabic data (ara.traineddata in ~/models/tessdata, see README).
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA = Path.home() / "models" / "tessdata"


def ocr(doc: pymupdf.Document) -> str:
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA)
    pages = []
    for page in list(doc)[:OCR_MAX_PAGES]:
        area = (page.rect.width / 72) * (page.rect.height / 72)
        img = Image.open(io.BytesIO(page.get_pixmap(dpi=min(DPI, int((MAX_PIXELS / area) ** 0.5))).tobytes("png")))
        pages.append(pytesseract.image_to_string(img, lang="ara+eng"))
    return "\n\f".join(pages)


def process(path: Path) -> dict:
    try:
        with pymupdf.open(path) as doc:
            n_pages = doc.page_count
            text = "\n\f".join(p.get_text("text") for p in doc)
            used_ocr = len(text.strip()) / max(n_pages, 1) < SCANNED_THRESHOLD
            if used_ocr:
                text = ocr(doc)
    except Exception as e:
        return {"attachment_id": path.stem, "error": str(e)[:200]}
    out = TEXT_AR / path.parent.name / f"{path.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return {"attachment_id": path.stem, "n_pages": n_pages, "n_chars": len(text), "ocr": used_ocr,
            "arabic_share": round(arabic_share(text), 3), "text_path": str(out.relative_to(TEXT_AR))}


def main():
    pdfs = sorted((PDF_DIR / "ar").glob("*/*.pdf"))
    with ProcessPoolExecutor(4) as pool:
        rows = list(tqdm(pool.map(process, pdfs), total=len(pdfs), desc="Arabic PDFs"))

    att = pd.read_csv(RAW / "attachments_ar.csv", parse_dates=["published"]).drop(columns=["language"])
    # Link each Arabic document to its English twin disclosure.
    twins = pd.read_csv(RAW / "disclosures.csv")[["id", "id_ar"]].rename(
        columns={"id": "disclosure_id_en", "id_ar": "disclosure_id"})
    docs = att.merge(pd.DataFrame(rows), on="attachment_id").merge(twins, on="disclosure_id", how="left")
    docs.to_csv(PROCESSED / "documents_ar.csv", index=False, encoding="utf-8-sig")
    ok = docs[docs["error"].isna()] if "error" in docs else docs
    print(f"{len(ok)} Arabic documents extracted ({int(ok['ocr'].sum())} via OCR)")
    print(f"median Arabic share of letters: {ok['arabic_share'].median():.2f}; "
          f"documents mostly Arabic (>=0.7): {(ok['arabic_share'] >= 0.7).mean():.0%}")


if __name__ == "__main__":
    main()
