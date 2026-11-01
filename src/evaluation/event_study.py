"""Event study - does text sentiment explain the price reaction?

For every event (a disclosure, a day of news, or a results call) we measure the
stock's abnormal return around it and test whether sentiment explains it.

Design choices (all chosen to avoid look-ahead bias):
  * Event day 0 = first trading session after publication. Items published
    before the 10:00 open count for that same day.
  * Abnormal return (AR) = stock return - equal-weighted DFM market return.
  * CAR[0,k] = sum of AR over days 0..k; CAR[-5,-1] checks for pre-event drift.
  * "Relative" sentiment compares an event with the same company's *earlier*
    events only (expanding / rolling windows that exclude the event itself).

Output: data/processed/events_disclosures.csv, events_news.csv, events_calls.csv
        results/event_study_summary.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import MARKET_INDEX, PRICE_DIR, PROCESSED, ROOT, STOCKS
from text_processing.doctype import doc_type
from collection.prices import clean_returns

MARKET_OPEN = pd.Timedelta(hours=10)
WINDOWS = {"CAR[0]": (0, 0), "CAR[0,1]": (0, 1), "CAR[0,2]": (0, 2), "CAR[0,4]": (0, 4), "CAR[-5,-1]": (-5, -1)}
MIN_HISTORY = 3  # earlier same-type events needed before a relative score is defined


# ---------------------------------------------------------------- prices ---
def abnormal_returns() -> dict[str, pd.Series]:
    market = pd.read_csv(MARKET_INDEX, parse_dates=["date"], index_col="date")["market_return"]
    out = {}
    for symbol in STOCKS:
        px = pd.read_csv(PRICE_DIR / f"{symbol}.csv", parse_dates=["date"], index_col="date")["Adj Close"]
        r = clean_returns(px)
        out[symbol] = (r - market.reindex(r.index)).dropna()
    return out


MAX_GAP = pd.Timedelta(days=7)


def event_day(ar: pd.Series, ts: pd.Timestamp) -> int | None:
    """Index (into ar) of the first trading session that could react to ts.

    Returns None for events before the stock's price history starts (e.g. news
    about DEWA before its 2022 listing) or with no session within MAX_GAP;
    otherwise all such events would pile onto the listing day.
    """
    cutoff = ts.normalize() if ts - ts.normalize() < MARKET_OPEN else ts.normalize() + pd.Timedelta(days=1)
    if cutoff < ar.index[0]:
        return None
    i = ar.index.searchsorted(cutoff)
    if i >= len(ar) or ar.index[i] - cutoff > MAX_GAP:
        return None
    return int(i)


def add_cars(events: pd.DataFrame, ars: dict[str, pd.Series], ts_col: str) -> pd.DataFrame:
    rows = []
    for e in events.itertuples(index=False):
        ar = ars[e.symbol]
        i = event_day(ar, getattr(e, ts_col))
        cars = {}
        for name, (a, b) in WINDOWS.items():
            ok = i is not None and i + a >= 0 and i + b < len(ar)
            cars[name] = ar.iloc[i + a:i + b + 1].sum() if ok else np.nan
        cars["day0"] = ar.index[i] if i is not None else pd.NaT
        rows.append(cars)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------- events ---
def disclosure_events() -> pd.DataFrame:
    docs = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["published"])
    sent = pd.read_csv(PROCESSED / "doc_sentiment.csv")
    d = docs.merge(sent, on="attachment_id").query("n_sentences > 0")
    d["doc_type"] = d["description"].map(doc_type)
    # One disclosure can have several attachments: sentence-weighted average.
    d["w"] = d["n_sentences"]
    ev = (d.assign(ws=d["score"] * d["w"], wneg=d["share_negative"] * d["w"])
          .groupby(["disclosure_id", "symbol", "published"])
          .agg(ws=("ws", "sum"), wneg=("wneg", "sum"), w=("w", "sum"),
               doc_type=("doc_type", lambda x: x.mode().iloc[0])).reset_index())
    ev["score"] = ev["ws"] / ev["w"]
    ev["share_negative"] = ev["wneg"] / ev["w"]
    ev = ev.drop(columns=["ws", "wneg", "w"]).sort_values("published")
    # Relative score: versus this company's earlier disclosures of the same type.
    grp = ev.groupby(["symbol", "doc_type"])["score"]
    prior_mean = grp.transform(lambda s: s.expanding().mean().shift())
    prior_n = grp.cumcount()
    ev["rel_score"] = (ev["score"] - prior_mean).where(prior_n >= MIN_HISTORY)
    return ev


def news_events() -> pd.DataFrame:
    n = pd.read_csv(PROCESSED / "news_sentiment.csv")
    n = n[n["names_company"]].copy()
    # Stored as Dubai time with offset; drop the offset after converting.
    n["published"] = pd.to_datetime(n["published"], utc=True).dt.tz_convert("Asia/Dubai").dt.tz_localize(None)
    n["date"] = n["published"].dt.normalize()
    ev = (n.groupby(["symbol", "date"])
          .agg(published=("published", "max"), score=("polarity", "mean"), n_headlines=("polarity", "size"),
               share_negative=("label", lambda x: (x == "negative").mean()))
          .reset_index().sort_values("date"))
    # Relative score: versus the stock's news over the previous 90 days (excluding today).
    parts = []
    for _, g in ev.groupby("symbol"):
        g = g.set_index("date")
        prior = g["score"].rolling("90D", closed="left").mean()
        count = g["score"].rolling("90D", closed="left").count()
        parts.append(g.assign(rel_score=(g["score"] - prior).where(count >= MIN_HISTORY)).reset_index())
    return pd.concat(parts, ignore_index=True)


def call_events() -> pd.DataFrame:
    tone = pd.read_csv(PROCESSED / "transcript_tone.csv", parse_dates=["call_date"])
    p = tone.pivot_table(index=["period", "call_date"], columns=["section", "role"], values="polarity")
    p.columns = [f"{s}_{r}" for s, r in p.columns]
    ev = p.reset_index().dropna(subset=["call_date"])
    ev["symbol"] = "EMIRATESNBD"
    # Call time of day is not in the transcript; calls are held after results, so
    # the conservative choice is to treat them as after-hours (next session).
    ev["published"] = ev["call_date"] + pd.Timedelta(hours=16)
    ev["score"] = ev.get("qa_analyst")
    ev["mgmt_minus_analyst"] = ev.get("prepared_management") - ev.get("qa_analyst")
    return ev


# ------------------------------------------------------------- statistics ---
def test(ev: pd.DataFrame, source: str, signal: str) -> list[dict]:
    rows = []
    for window in WINDOWS:
        d = ev[[signal, window]].dropna()
        if len(d) < 10:
            continue
        rho, p_rho = stats.spearmanr(d[signal], d[window])
        ols = sm.OLS(d[window], sm.add_constant(d[signal])).fit(cov_type="HC1")
        lo, hi = d[signal].quantile([1 / 3, 2 / 3])
        bottom, top = d.loc[d[signal] <= lo, window], d.loc[d[signal] >= hi, window]
        t, p_t = stats.ttest_ind(top, bottom, equal_var=False)
        rows.append({"source": source, "signal": signal, "window": window, "n": len(d),
                     "spearman_rho": rho, "spearman_p": p_rho,
                     "ols_beta": ols.params[signal], "ols_p": ols.pvalues[signal],
                     "CAR_top_tercile_%": top.mean() * 100, "CAR_bottom_tercile_%": bottom.mean() * 100,
                     "top_minus_bottom_%": (top.mean() - bottom.mean()) * 100, "t_test_p": p_t})
    return rows


def main():
    ars = abnormal_returns()
    results = []

    disc = add_cars(disclosure_events(), ars, "published")
    disc.to_csv(PROCESSED / "events_disclosures.csv", index=False)
    for sig in ("score", "rel_score", "share_negative"):
        results += test(disc, "DFM disclosures (all)", sig)
    pr = disc[disc["doc_type"] == "press_release"]
    for sig in ("score", "rel_score"):
        results += test(pr, "DFM press releases", sig)

    news = add_cars(news_events(), ars, "published")
    news.to_csv(PROCESSED / "events_news.csv", index=False)
    for sig in ("score", "rel_score", "share_negative"):
        results += test(news, "Google News (stock-days)", sig)

    calls = add_cars(call_events(), ars, "published")
    calls.to_csv(PROCESSED / "events_calls.csv", index=False)
    for sig in ("qa_analyst", "prepared_management", "mgmt_minus_analyst"):
        if sig in calls:
            results += test(calls, "ENBD results calls", sig)

    res = pd.DataFrame(results)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "event_study_summary.csv", index=False)

    print(f"Events: {disc['CAR[0,1]'].notna().sum()} disclosures, {news['CAR[0,1]'].notna().sum()} news stock-days, "
          f"{calls['CAR[0,1]'].notna().sum()} calls")
    pd.set_option("display.width", 250)
    cols = ["source", "signal", "window", "n", "spearman_rho", "spearman_p", "CAR_top_tercile_%",
            "CAR_bottom_tercile_%", "top_minus_bottom_%", "t_test_p"]
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
