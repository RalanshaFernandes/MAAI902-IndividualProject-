"""Project-wide settings: paths, the stock universe and data-collection parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
# Raw PDFs (~1.5 GB). Override with the DFM_PDF_DIR environment variable to keep
# them elsewhere, e.g. outside a cloud-synced folder.
PDF_DIR = Path(os.environ.get("DFM_PDF_DIR", RAW / "pdfs"))
PRICE_DIR = RAW / "prices"
# Everything the pipeline generates lives under data/processed/.
PROCESSED = DATA / "processed"
TEXT_DIR = PROCESSED / "text_en"
TEXT_AR_DIR = PROCESSED / "text_ar"
MARKET_INDEX = PROCESSED / "market_index.csv"

# DFM symbol -> (Yahoo Finance ticker, company name, sector)
STOCKS = {
    "EMAAR": ("EMAAR.AE", "Emaar Properties", "Real Estate"),
    "EMIRATESNBD": ("EMIRATESNBD.AE", "Emirates NBD", "Banking"),
    "DIB": ("DIB.AE", "Dubai Islamic Bank", "Banking"),
    "DU": ("DU.AE", "Emirates Integrated Telecommunications (du)", "Telecom"),
    "DEWA": ("DEWA.AE", "Dubai Electricity & Water Authority", "Utilities"),
    "SALIK": ("SALIK.AE", "Salik", "Transport"),
}

# Price history and PDF downloads start here; older disclosures keep metadata only.
START_DATE = "2019-01-01"

# DFM endpoints, taken from the public runtime config of www.dfm.ae.
EFSAH_API = "https://api2.dfm.ae/efsah/v1"
DOCS_BASE = "https://feeds.dfm.ae/documents"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (MAAI902 student research project)",
    "Origin": "https://www.dfm.ae",
    "Referer": "https://www.dfm.ae/",
}

# Seconds to wait between requests so we do not load DFM's servers.
REQUEST_DELAY = 1.0
