"""Download daily prices from Yahoo Finance and clean them.

Run this module to download daily OHLCV for each stock.
Output: data/raw/prices/<SYMBOL>.csv

clean_returns() is shared by the market index, the event study and the signal.
Yahoo's DFM data contains occasional one-day bad ticks: e.g. on 2020-02-02
DIB closed at 1.30 and du at 0.185 (instead of ~5.7), then both returned to
normal the next day. Left in, a single such row creates +300% / +2,900%
"returns" that swamp any real effect.

A row is treated as a bad tick when the price jumps by more than SPIKE and the
next day jumps back by more than SPIKE in the opposite direction, ending
within REVERT of the price before the spike.
"""
import pandas as pd

from config import PRICE_DIR, START_DATE, STOCKS

SPIKE = 0.20
REVERT = 0.10


def remove_spikes(close: pd.Series) -> pd.Series:
    """Return the price series with one-day reversing spikes removed (NaN)."""
    c = close.copy()
    prev, nxt = c.shift(1), c.shift(-1)
    up, back = c / prev - 1, nxt / c - 1
    bad = (up.abs() > SPIKE) & (back.abs() > SPIKE) & (up * back < 0) & ((nxt / prev - 1).abs() < REVERT)
    c[bad] = float("nan")
    return c


def clean_returns(close: pd.Series) -> pd.Series:
    """Daily returns after removing bad ticks; gaps are bridged (t-1 -> t+1)."""
    return remove_spikes(close).dropna().pct_change().dropna()


def main():
    import yfinance as yf  # only the download needs it, not the modules using clean_returns

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, (ticker, _, _) in STOCKS.items():
        df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
        # yfinance returns (field, ticker) column pairs; keep just the field names.
        df.columns = df.columns.get_level_values(0)
        # Drop the empty rows Yahoo emits around the UAE's 2022 weekend change.
        df = df.dropna(how="all")
        df.index.name = "date"
        df.to_csv(PRICE_DIR / f"{symbol}.csv")
        print(f"{symbol:12} {len(df):5} rows  {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
