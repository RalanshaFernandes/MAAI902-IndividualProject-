"""Collect independent news headlines from Google News search feeds.

Company disclosures are written by the companies themselves and lean positive.
Independent media headlines give an outside view of the same events.

Google News RSS supports `after:`/`before:` date operators, so we query one
calendar month at a time. Only headline, outlet, date and link are stored, not
article bodies, which respects publishers' copyright.

Each month's raw result is cached in data/raw/news/<SYMBOL>/<YYYY-MM>.json, so
the script can be stopped and resumed.
Output: data/processed/news.csv (one row per unique headline per stock)
"""
import json
import re
import time
from datetime import date
from urllib.parse import quote

import feedparser
import pandas as pd
import requests
from tqdm import tqdm

from config import PROCESSED, RAW, START_DATE

NEWS_DIR = RAW / "news"
DELAY = 2.0  # seconds between requests

# Search query per stock, plus the words that count as "the headline names the company".
QUERIES = {
    "EMAAR": ('"Emaar"', r"\bemaar\b"),
    "EMIRATESNBD": ('"Emirates NBD"', r"emirates\s?nbd|\benbd\b"),
    "DIB": ('"Dubai Islamic Bank"', r"dubai islamic bank|\bdib\b"),
    # "du" alone is too short to search for; "Salik" alone also matches people
    # named Salik. These versions were compared on a sample month (Feb 2024).
    "DU": ("du telecom UAE", r"\bdu\b|emirates integrated|\beitc\b"),
    "DEWA": ('"DEWA" OR "Dubai Electricity and Water Authority"', r"\bdewa\b|dubai electricity"),
    "SALIK": ('"Salik" Dubai', r"\bsalik\b"),
}


# Different companies that share a name with one of ours (entity disambiguation).
# "Emaar The Economic City" is a separate Saudi-listed company, whereas Emaar
# Malls / Emaar Development / Emaar Misr are subsidiaries of Emaar Properties and stay.
EXCLUDE = {
    "EMAAR": r"economic city|emaar ec\b|\beec\b|kaec|king abdullah economic",
    # Indonesian football club / esports team / band, a Cambodian club, a Mexican developer.
    "DEWA": r"dewa united|kompong dewa|dewa 19|dewa fc|esports|\bmpl\b|cruz azul|developer dewa",
}


def month_windows(start: str) -> list[tuple[str, str]]:
    """[(first day of month, first day of next month), ...] up to today."""
    starts = pd.date_range(start, date.today(), freq="MS")
    return [(s.strftime("%Y-%m-%d"), (s + pd.offsets.MonthBegin()).strftime("%Y-%m-%d")) for s in starts]


def fetch(query: str, after: str, before: str) -> list[dict]:
    q = quote(f"{query} after:{after} before:{before}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=AE&ceid=AE:en"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return [{"title": e.title, "source": e.get("source", {}).get("title", ""),
             "published": e.get("published", ""), "link": e.link}
            for e in feedparser.parse(resp.content).entries]


def collect(symbol: str, query: str) -> list[dict]:
    rows = []
    for after, before in month_windows(START_DATE):
        cache = NEWS_DIR / symbol / f"{after[:7]}.json"
        if cache.exists():
            items = json.loads(cache.read_text(encoding="utf-8"))
        else:
            items = fetch(query, after, before)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            time.sleep(DELAY)
        if len(items) >= 90:
            print(f"  {symbol} {after[:7]}: {len(items)} results, feed may be truncated")
        rows += [{**i, "symbol": symbol} for i in items]
    return rows


def clean_title(title: str, source: str) -> str:
    """Google appends ' - Outlet Name' to every headline; remove it."""
    if source and title.endswith(f" - {source}"):
        title = title[: -len(source) - 3]
    return re.sub(r"\s+", " ", title).strip()


def main():
    rows = []
    for symbol, (query, _) in tqdm(QUERIES.items(), desc="Stocks"):
        rows += collect(symbol, query)

    news = pd.DataFrame(rows)
    news["headline"] = [clean_title(t, s) for t, s in zip(news["title"], news["source"])]
    news["published"] = pd.to_datetime(news["published"], utc=True, errors="coerce").dt.tz_convert("Asia/Dubai")
    news = news.dropna(subset=["published"])
    # The same story is often syndicated by several outlets; keep its first appearance.
    news["key"] = news["headline"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    news = news.sort_values("published").drop_duplicates(["symbol", "key"])
    news["names_company"] = [bool(re.search(QUERIES[s][1], h, re.I)) and
                             not (s in EXCLUDE and re.search(EXCLUDE[s], h, re.I))
                             for s, h in zip(news["symbol"], news["headline"])]

    cols = ["symbol", "published", "headline", "source", "names_company", "link"]
    news[cols].sort_values(["symbol", "published"]).to_csv(PROCESSED / "news.csv", index=False, encoding="utf-8-sig")

    print(f"\n{len(news)} unique headlines")
    print(news.groupby("symbol").agg(headlines=("headline", "size"), names_company=("names_company", "mean"))
          .round(2).to_string())
    print("\nTop outlets:\n", news["source"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
