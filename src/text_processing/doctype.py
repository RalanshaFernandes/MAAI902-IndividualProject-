"""Rule-based document-type labels from DFM attachment descriptions / file names.

Companies encode the document type in file names (e.g. "DEWA_PR_E_...",
"DIB FS Q3", "EmiratesNBD_NOT_..."), so simple keyword rules are a strong
baseline. Rules are checked in order; the first match wins.
"""
import re

RULES = [
    ("press_release", r"press|_pr_|_pr\b|\bpr\b|prl|\ber\b"),
    ("financial_statements", r"_fs|\bfs\b|financial statement|interim|consolidated"),
    ("board_meeting", r"bod|board"),
    ("general_assembly", r"general|agm|assembly|election"),
    ("dividend", r"dividend|_div"),
    ("rating", r"rating"),
    ("disclosure_form", r"disclosure|form|notification|_not_|notice"),
]


def doc_type(description: str) -> str:
    text = str(description).lower()
    return next((name for name, pattern in RULES if re.search(pattern, text)), "other")
