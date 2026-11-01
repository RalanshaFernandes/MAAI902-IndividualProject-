"""Build an equal-weighted DFM market index as the event-study benchmark.

Yahoo Finance has no history for the DFM General Index, so we take the list
of DFM-listed equities from DFM's own API, download each one's daily prices
from Yahoo (".AE" suffix), and average their daily returns.

To limit noise from thinly traded shares, a stock only enters the index on
days when it traded (volume > 0), and the index needs at least 10 stocks.

Output: data/processed/market_index.csv (date, market_return, n_stocks)
"""
import json

import pandas as pd
import requests
import yfinance as yf

from config import HEADERS, MARKET_INDEX, RAW, START_DATE
from collection.dfm_api import _get
from collection.prices import clean_returns

WIDGETS_API = "https://api2.dfm.ae/web/widgets/v1/data"
EQUITIES_CACHE = RAW / "dfm_equities.json"
FALLBACK_DISCLOSURES = 3000  # roughly the last few months of filings across all issuers
MIN_STOCKS = 10


def dfm_equities() -> list[str]:
    """DFM equity symbols from the securities-list widget, cached in data/raw/dfm_equities.json.

    If the widget does not respond (it hung for minutes in October 2026), fall back to
    the issuers of the most recent disclosures: every listed company files regularly.
    """
    try:
        resp = requests.post(WIDGETS_API, headers=HEADERS, timeout=60,
                             data={"Command": "LiteSecuritiesLists", "Language": "en", "lang": "en",
                                   "securitytype": "equities"})
        resp.raise_for_status()
        symbols = sorted({r["SecuritySymbol"] for r in resp.json() if r.get("Exchange") == "DFM"})
    except requests.RequestException as e:
        print(f"Securities list unavailable ({e.__class__.__name__}); using recent disclosure issuers")
        try:
            symbols = sorted({r["issuer_symbol"] for skip in range(0, FALLBACK_DISCLOSURES, 20)
                              for r in _get("prototype_efsah", lang="en", symbol="", take=20, skip=skip)
                              if r.get("issuer_symbol")})
        except requests.RequestException:
            if not EQUITIES_CACHE.exists():
                raise
            print(f"Disclosure API unavailable too; using cached list {EQUITIES_CACHE.name}")
            return json.loads(EQUITIES_CACHE.read_text(encoding="utf-8"))
    EQUITIES_CACHE.write_text(json.dumps(symbols), encoding="utf-8")
    return symbols


def main():
    symbols = dfm_equities()
    print(f"{len(symbols)} DFM equities listed")
    data = yf.download([f"{s}.AE" for s in symbols], start=START_DATE, progress=False,
                       auto_adjust=True, group_by="column", threads=True)
    close, volume = data["Close"], data["Volume"]
    available = close.columns[close.notna().sum() > 100]
    print(f"{len(available)} have usable price history on Yahoo")

    # Remove one-day bad ticks per stock, then compute returns across the gaps.
    returns = pd.DataFrame({s: clean_returns(close[s].dropna()) for s in available})
    returns = returns.where(volume[available].reindex(returns.index) > 0)  # ignore days a stock did not trade
    returns = returns.where(returns.abs() < 0.5)            # drop any remaining obvious data errors
    market = pd.DataFrame({"market_return": returns.mean(axis=1), "n_stocks": returns.notna().sum(axis=1)})
    market = market[market["n_stocks"] >= MIN_STOCKS]
    market.index.name = "date"
    MARKET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(MARKET_INDEX)
    print(f"Index: {len(market)} days, {market.index.min().date()} -> {market.index.max().date()}, "
          f"median {market['n_stocks'].median():.0f} stocks per day")


if __name__ == "__main__":
    main()
