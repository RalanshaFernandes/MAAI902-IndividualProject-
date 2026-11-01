"""Collect Emirates NBD results-call transcripts and split them into speaker turns.

Emirates NBD is the only company in our set that publishes call transcripts
(checked the other five companies' IR sites, web search and DFM filings; their
"Result of Earnings Call" filings are slide decks, not transcripts).

A call has two parts with very different tone:
  - prepared remarks: scripted by management, typically upbeat
  - Q&A: analysts probe weak spots (provisions, margins, guidance)
Splitting by speaker lets us compare management tone with analyst tone.

Transcript layout (stable since 2017):
  CORPORATE PARTICIPANTS
  Shayne Nelson – Emirates NBD – Group CEO       <- management names
  ...
  Operator                                       <- speaker label on its own line
  Ladies and gentlemen, welcome ...
  QUESTIONS AND ANSWERS                          <- start of Q&A
  Shabbir Malik - EFG Hermes                     <- analyst label "Name - Firm"

Output: data/raw/transcripts/EMIRATESNBD/<qN_YYYY>.pdf
        data/processed/transcript_turns.csv (one row per speaker turn)
"""
import difflib
import re
import time
from urllib.parse import urljoin

import pandas as pd
import pymupdf
import requests

from config import PROCESSED, RAW, REQUEST_DELAY

IR_PAGE = "https://www.emiratesnbd.com/en/investor-relations"
OUT_DIR = RAW / "transcripts" / "EMIRATESNBD"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

MGMT_LINE = re.compile(r"^(.+?)\s+[–-]\s+Emirates NBD\s+[–-]", re.I)
ANALYST_LABEL = re.compile(r"^[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){0,3}\s+[–-]\s+[A-Za-z&.,' ]{2,40}$")
# Newer transcripts (from late 2024) drop the participant list, the Q&A heading
# and analysts' firms: a speaker label is then just a name on its own line.
NAME_LABEL = re.compile(r"^[A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-.]*){1,3}$")
# From late 2025 (NetRoadshow format) labels look like "Shayne Nelson (Group CEO)",
# "Adam (Moderator)" or "Jon Peace (UBS)".
PAREN_LABEL = re.compile(r"^([A-Z][A-Za-z'\-. ]{1,40}?) \(([^()]{2,60})\)$")
QA_MARKER = re.compile(r"^QUESTIONS?\s+AND\s+ANSWERS?$", re.I)
DATE = re.compile(r"\b(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d)\b")
ISO_DATE = re.compile(r"Document created on: (20\d\d-\d\d-\d\d)")
US_DATE = re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
                     r" \d{1,2},\s*20\d\d)\b")  # "October 23,\n2025"


def transcript_links() -> dict[str, str]:
    """Map 'q1_2025' -> absolute PDF URL, from the IR page."""
    html = requests.get(IR_PAGE, headers=HEADERS, timeout=60).text
    links = {}
    for href in re.findall(r'href="([^"]+transcript[^"]*\.pdf)"', html, re.I):
        m = re.search(r"(q\d)_(20\d\d)", href.lower())
        if m:
            links.setdefault(f"{m.group(1)}_{m.group(2)}", urljoin(IR_PAGE, href))
    return links


def download(links: dict[str, str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for period, url in sorted(links.items()):
        out = OUT_DIR / f"{period}.pdf"
        if out.exists():
            continue
        resp = requests.get(url, headers=HEADERS, timeout=120)
        if resp.ok and resp.content.startswith(b"%PDF"):
            out.write_bytes(resp.content)
        else:
            print(f"  failed {period}: HTTP {resp.status_code}")
        time.sleep(REQUEST_DELAY)


def management_names(texts: list[str]) -> set[str]:
    """Every executive listed under CORPORATE PARTICIPANTS in any transcript."""
    return {m.group(1).strip() for t in texts for l in t.splitlines() if (m := MGMT_LINE.match(l.strip()))}


def is_management(name: str, management: set[str]) -> bool:
    # Fuzzy match tolerates typos in the transcripts, e.g. "Patrick Sullilvan".
    return any(difflib.SequenceMatcher(None, name.lower(), m.lower()).ratio() >= 0.85 for m in management)


def split_turns(text: str, management: set[str]) -> tuple[str | None, list[dict]]:
    """Split one transcript into speaker turns tagged with role and section."""
    lines = [l.strip() for l in text.splitlines()]
    m = DATE.search(text) or US_DATE.search(text) or ISO_DATE.search(text)
    call_date = re.sub(r"\s+", " ", m.group(1)) if m else None
    has_qa_marker = any(QA_MARKER.match(l) for l in lines)

    turns, speaker, role, section, buf = [], None, None, "prepared", []

    def flush():
        if speaker and buf:
            turns.append({"speaker": speaker, "role": role, "section": section, "text": " ".join(buf)})

    started = False
    for line in lines:
        if not line or MGMT_LINE.match(line) or line.upper() in ("CORPORATE PARTICIPANTS", "END"):
            continue
        if QA_MARKER.match(line):
            flush(); speaker, buf, section = None, [], "qa"
            continue
        paren = PAREN_LABEL.match(line)
        if paren:
            name, tag = paren.group(1).strip(), paren.group(2)
            line = name
            new_role = ("operator" if re.search(r"moderator|operator", tag, re.I) else
                        "management" if is_management(name, management) or
                        re.search(r"\b(CEO|CFO|Chief|Head|Investor Relations)\b", tag) else "analyst")
        elif line == "Operator":
            new_role = "operator"
        elif (NAME_LABEL.match(line) or ANALYST_LABEL.match(line)) and is_management(line, management):
            new_role = "management"
        elif started and (ANALYST_LABEL.match(line) or NAME_LABEL.match(line)):
            new_role = "analyst"
        else:
            new_role = None

        if new_role:
            flush()
            # Without a Q&A heading, the Q&A starts at the first analyst turn.
            if new_role == "analyst" and not has_qa_marker:
                section = "qa"
            speaker, role, buf, started = line, new_role, [], True
        elif started:
            buf.append(line)
    flush()
    return call_date, turns


def main():
    links = transcript_links()
    print(f"{len(links)} transcripts listed on the IR page")
    download(links)

    texts = {}
    for pdf in sorted(OUT_DIR.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            texts[pdf.stem] = "\n".join(p.get_text() for p in doc)
    management = management_names(list(texts.values()))
    print("management speakers:", sorted(management))

    rows = []
    for stem, text in texts.items():
        call_date, turns = split_turns(text, management)
        if not turns:
            print(f"  no speaker turns found in {stem}")
        q, year = stem.split("_")
        for i, t in enumerate(turns):
            rows.append({"symbol": "EMIRATESNBD", "period": f"{year}-{q.upper()}", "call_date": call_date,
                         "turn": i, **t, "n_words": len(t["text"].split())})

    df = pd.DataFrame(rows)
    df["call_date"] = pd.to_datetime(df["call_date"], format="mixed", dayfirst=True, errors="coerce")
    # Transcripts mix en dashes and hyphens in labels ("Name – Firm" / "Name - Firm").
    df["speaker"] = df["speaker"].str.replace("–", "-").str.replace(r"\s+", " ", regex=True)
    df.to_csv(PROCESSED / "transcript_turns.csv", index=False, encoding="utf-8-sig")

    print(f"{df['period'].nunique()} calls, {len(df)} speaker turns")
    print(df.groupby(["section", "role"])["n_words"].agg(["count", "sum"]).to_string())
    missing_qa = set(df["period"]) - set(df.loc[df["section"] == "qa", "period"])
    if missing_qa:
        print("calls with no Q&A section detected:", sorted(missing_qa))


if __name__ == "__main__":
    main()
