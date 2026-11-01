"""OCR the scanned PDFs (no text layer) with Tesseract.

Only the first OCR_MAX_PAGES pages of each document are processed: scanned
disclosures are mostly 1-2 page signed letters, and in longer documents the
substantive content (summary, key figures) comes first.

Updates: data/processed/text_en/<SYMBOL>/<id>.txt (overwritten with OCR text)
         data/processed/documents.csv (adds `ocr` flag, refreshes size/language columns)
"""
import io
from concurrent.futures import ProcessPoolExecutor

import pymupdf as fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm

from config import PDF_DIR, PROCESSED, TEXT_DIR
from text_processing.extract_text_en import SCANNED_THRESHOLD, language_of

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_MAX_PAGES = 5
DPI = 300  # Tesseract accuracy drops sharply below ~300 dpi
WORKERS = 4


MAX_PIXELS = 40_000_000  # some scans have huge page sizes; cap the rendered image


def ocr_pdf(args: tuple[str, str]) -> dict:
    symbol, attachment_id = args
    pages = []
    try:
        with fitz.open(PDF_DIR / symbol / f"{attachment_id}.pdf") as doc:
            for page in list(doc)[:OCR_MAX_PAGES]:
                # Lower the dpi for oversized pages so the image stays under MAX_PIXELS.
                area_in2 = (page.rect.width / 72) * (page.rect.height / 72)
                dpi = min(DPI, int((MAX_PIXELS / area_in2) ** 0.5))
                img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png")))
                pages.append(pytesseract.image_to_string(img, lang="eng"))
    except Exception as e:
        print(f"OCR failed for {symbol}/{attachment_id}: {e}")
        return {"attachment_id": attachment_id, "text": "", "pages_ocr": 0}
    text = "\n\f".join(pages)
    (TEXT_DIR / symbol / f"{attachment_id}.txt").write_text(text, encoding="utf-8")
    return {"attachment_id": attachment_id, "text": text, "pages_ocr": len(pages)}


def main():
    docs = pd.read_csv(PROCESSED / "documents.csv")
    # Skip documents OCR'd in an earlier run, so a data refresh only OCRs new scans.
    docs["ocr"] = docs["ocr"].fillna(False).astype(bool) if "ocr" in docs else False
    todo = docs[docs["scanned"].fillna(False).astype(bool) & ~docs["ocr"]]
    jobs = list(zip(todo["symbol"], todo["attachment_id"]))

    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(tqdm(pool.map(ocr_pdf, jobs), total=len(jobs), desc="OCR"))

    for r in results:
        i = docs.index[docs["attachment_id"] == r["attachment_id"]][0]
        lang, ratio = language_of(r["text"])
        cpp = len(r["text"].strip()) / max(r["pages_ocr"], 1)
        docs.loc[i, ["n_chars", "chars_per_page", "arabic_ratio", "language", "ocr"]] = \
            [len(r["text"]), round(cpp), ratio, lang, True]
        # Still unreadable after OCR (blank pages, signatures only, stamps).
        docs.loc[i, "scanned"] = cpp < SCANNED_THRESHOLD

    docs.to_csv(PROCESSED / "documents.csv", index=False, encoding="utf-8-sig")
    ocr = docs[docs["ocr"]]
    print(f"\nOCR'd {len(ocr)} documents; {(~ocr['scanned']).sum()} now readable, "
          f"{ocr['scanned'].sum()} still unreadable")
    print(f"Readable documents overall: {(~docs['scanned']).sum()} of {len(docs)}")


if __name__ == "__main__":
    main()
