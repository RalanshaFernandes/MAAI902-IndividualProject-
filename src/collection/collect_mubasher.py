"""Collect company news from Mubasher (English and Arabic).

Mubasher is a Gulf financial news service with a news page per DFM stock.
Each item has a date, headline, link and a lead paragraph, so it gives more
text per story than Google News headlines. Only these fields are stored (not
full articles).

Pages are cached in data/raw/mubasher/<lang>/<SYMBOL>/<page>.html, so the
script can be stopped and resumed.
Output: data/processed/mubasher.csv
"""
import html
import re
import time

import pandas as pd
import requests

from config import PROCESSED, RAW, START_DATE, STOCKS

SITES = {"en": "https://english.mubasher.info", "ar": "https://www.mubasher.info"}
CACHE = RAW / "mubasher"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
DELAY = 2.0
MAX_PAGES = 60  # safety stop

ITEM = re.compile(
    r'mi-article-media-block__date">\s*([^<]+?)\s*</span>.*?'
    r'mi-article-media-block__title" href="([^"]+)">\s*([^<]+?)\s*</a>'
    r'(?:\s*<div[^>]*mi-article-media-block__text[^>]*>\s*([^<]*?)\s*</div>)?',
    re.S)
# Arabic month names -> month number, for the Arabic site's dates.
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
AR_MONTHS = {"يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "إبريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7,
             "أغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}


def parse_date(s: str, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Parse Mubasher's relative-style dates.

    The site shows "13 September 2021 03:26 PM" for older items, drops the year
    for the current year ("11 May 11:54 AM") and shows only the time for today
    ("06:00 PM"). Arabic pages use Arabic month names and ص/م for AM/PM.
    """
    today = today or pd.Timestamp.now().normalize()
    s = s.strip()
    for ar, num in AR_MONTHS.items():
        s = s.replace(ar, MONTH_NAMES[num - 1])
    s = s.replace("ص", "AM").replace("م", "PM").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    s = re.sub(r"\s+", " ", s)

    if re.fullmatch(r"\d{1,2}:\d{2} [AP]M", s):                       # time only -> today
        return pd.to_datetime(f"{today:%d %B %Y} {s}", format="%d %B %Y %I:%M %p")
    if re.fullmatch(r"\d{1,2} [A-Za-z]+ \d{1,2}:\d{2} [AP]M", s):     # no year -> this year
        day_month, time_ = s.rsplit(" ", 2)[0], " ".join(s.rsplit(" ", 2)[1:])
        ts = pd.to_datetime(f"{day_month} {today.year} {time_}", format="%d %B %Y %I:%M %p", errors="coerce")
        # A "future" date means the item is from last December etc.
        return ts - pd.DateOffset(years=1) if pd.notna(ts) and ts > today + pd.Timedelta(days=1) else ts
    return pd.to_datetime(s, format="%d %B %Y %I:%M %p", errors="coerce")


def fetch_page(lang: str, symbol: str, page: int) -> str:
    cache = CACHE / lang / symbol / f"{page}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = f"{SITES[lang]}/markets/DFM/stocks/{symbol}/news" + (f"/{page}" if page > 1 else "")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(DELAY)
    return resp.text


def collect(lang: str, symbol: str) -> list[dict]:
    rows, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        items = ITEM.findall(fetch_page(lang, symbol, page))
        new = [i for i in items if i[1] not in seen]
        if not new:  # past the last page, the site repeats or returns nothing
            break
        for date, link, title, lead in new:
            seen.add(link)
            rows.append({"lang": lang, "symbol": symbol, "published": parse_date(date),
                         "headline": html.unescape(title), "lead": html.unescape(lead or ""),
                         "link": SITES[lang] + link})
        oldest = min((r["published"] for r in rows[-len(new):] if pd.notna(r["published"])), default=None)
        if oldest is not None and oldest < pd.Timestamp(START_DATE):
            break
    return rows


def main():
    rows = []
    for lang in SITES:
        for symbol in STOCKS:
            try:
                got = collect(lang, symbol)
            except requests.HTTPError as e:
                print(f"{lang} {symbol}: {e}")
                continue
            print(f"{lang} {symbol:12} {len(got):4} items")
            rows += got

    df = pd.DataFrame(rows)
    df = df[df["published"] >= START_DATE].drop_duplicates(["lang", "symbol", "link"])
    df.sort_values(["lang", "symbol", "published"]).to_csv(PROCESSED / "mubasher.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} items since {START_DATE}")
    print(df.groupby(["lang", "symbol"]).agg(items=("headline", "size"), oldest=("published", "min")).to_string())


if __name__ == "__main__":
    main()
