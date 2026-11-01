"""Download disclosure PDFs published on or after START_DATE.

English (default): every English attachment -> <PDF_DIR>/<SYMBOL>/<attachment_id>.pdf
Arabic (--lang ar): attachments of the Arabic twins of English disclosures,
    i.e. the same disclosure filed in Arabic -> <PDF_DIR>/ar/<SYMBOL>/<attachment_id>.pdf
    These parallel documents let us compare English and Arabic sentiment models
    on the same content without needing Arabic labels.

Existing files are skipped, so the script can be stopped and resumed safely.
Output log: data/processed/download_log[_ar].csv
"""
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

from config import DOCS_BASE, HEADERS, PDF_DIR, PROCESSED, RAW, REQUEST_DELAY, START_DATE


def attachments(lang: str) -> tuple[pd.DataFrame, Path]:
    if lang == "en":
        att = pd.read_csv(RAW / "attachments.csv", parse_dates=["published"])
        return att[att["published"] >= START_DATE], PDF_DIR
    # Arabic: only twins of English disclosures in the study period.
    disc = pd.read_csv(RAW / "disclosures.csv", parse_dates=["published"])
    twins = disc.loc[(disc["published"] >= START_DATE) & disc["id_ar"].notna(), "id_ar"]
    att = pd.read_csv(RAW / "attachments_ar.csv", parse_dates=["published"])
    return att[att["disclosure_id"].isin(twins)], PDF_DIR / "ar"


def main(lang: str = "en"):
    att, out_dir = attachments(lang)
    session = requests.Session()
    session.headers.update(HEADERS)
    log = []

    for row in tqdm(att.itertuples(), total=len(att), desc=f"PDFs ({lang})"):
        out = out_dir / row.symbol / f"{row.attachment_id}.pdf"
        if out.exists() and out.stat().st_size > 0:
            log.append((row.attachment_id, "cached", out.stat().st_size))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = session.get(DOCS_BASE + quote(row.r_path), timeout=120)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            out.write_bytes(resp.content)
            log.append((row.attachment_id, "ok", len(resp.content)))
        except Exception as e:
            log.append((row.attachment_id, f"error: {e}"[:200], 0))
        time.sleep(REQUEST_DELAY)

    log = pd.DataFrame(log, columns=["attachment_id", "status", "bytes"])
    log.to_csv(PROCESSED / f"download_log{'' if lang == 'en' else '_' + lang}.csv", index=False)
    print(log["status"].str.split(":").str[0].value_counts().to_string())
    print(f"Total size: {log['bytes'].sum() / 1e6:.0f} MB in {out_dir}")


if __name__ == "__main__":
    main("ar" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "ar" else "en")
