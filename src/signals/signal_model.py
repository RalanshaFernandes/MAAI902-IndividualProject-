"""Transparent Buy / Hold / Sell signal and backtest.

The event study showed that text sentiment alone barely predicts short-term
returns, so the signal is a simple, explainable scoring rule rather than a
fitted ML model (which would overfit ~1,700 days x 6 stocks of noisy data).

Daily, per stock, using only information available at that day's close:
  news      = recent company-named headline tone (EWM, half-life 10 sessions)
  disclosure= recent tone of substantive company disclosures (EWM, half-life 20)
  momentum  = sum of abnormal returns over the last 60 sessions
Each component is turned into a z-score against its own *previous* 252 sessions,
clipped to [-2, 2] and scaled to [-1, 1]. A text component with no recent items
is missing (NaN), not zero: "no information" is not "neutral information".

  score     = equal-weighted mean of the components that are available
  BUY if score > +THRESHOLD, SELL if score < -THRESHOLD, else HOLD
  strength  = |score| shrunk when text coverage is thin or volatility is high,
              reported as Low / Medium / High. Deliberately NOT called a
              probability or "confidence": back-tests show BUY signals were
              right only about half the time, so a % would mislead users.
  agreement = how many available components point the same way as the signal

Backtests (signal at close t, position from t+1, abnormal returns vs DFM index):
  * forward 5/20-day abnormal return by signal class
  * long-short on abnormal returns: BUY = +1, SELL = -1, HOLD = 0, cost per position change
Variants compare price-only, sentiment-only and combined signals: does NLP add value?

Output: data/processed/signals.csv, results/signal_classes.csv, results/backtest_summary.csv,
        results/backtest_robustness.csv (robustness check; `python -m signals.signal_model --robustness` for only that)
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import MARKET_INDEX, PRICE_DIR, PROCESSED, ROOT, STOCKS
from collection.prices import clean_returns

THRESHOLD = 0.25
STRENGTH_BANDS = (0.25, 0.5)   # Low < 0.25 (HOLD territory) <= Medium < 0.5 <= High
AGREE_MIN = 0.15               # a component "points somewhere" only beyond +/-0.15
Z_WINDOW = 252
COST = 0.002          # 0.2% per unit of position change (commission + spread, approx.)
SUBSTANTIVE = {"press_release", "financial_statements", "rating", "dividend", "other"}
VARIANTS = {"price_only": ["momentum"], "sentiment_only": ["news", "disclosure"],
            "combined": ["news", "disclosure", "momentum"]}


def rolling_z(s: pd.Series) -> pd.Series:
    """z-score against the previous Z_WINDOW sessions only (no look-ahead), scaled to [-1, 1]."""
    past = s.shift(1).rolling(Z_WINDOW, min_periods=60)
    return ((s - past.mean()) / past.std()).clip(-2, 2) / 2


def text_component(events: pd.DataFrame, index: pd.DatetimeIndex, halflife: int) -> tuple[pd.Series, pd.Series]:
    """Daily EWM of event scores on their reaction day (day0), plus a 30-session item count."""
    daily = events.groupby("day0")["score"].mean().reindex(index)
    count = events.groupby("day0").size().reindex(index, fill_value=0)
    ewm = daily.ewm(halflife=halflife, ignore_na=False).mean()
    recent = count.rolling(30, min_periods=1).sum()
    return ewm.where(recent > 0), recent


def build_signals() -> pd.DataFrame:
    market = pd.read_csv(MARKET_INDEX, parse_dates=["date"], index_col="date")["market_return"]
    news = pd.read_csv(PROCESSED / "events_news.csv", parse_dates=["day0"]).dropna(subset=["day0"])
    disc = pd.read_csv(PROCESSED / "events_disclosures.csv", parse_dates=["day0"]).dropna(subset=["day0"])
    disc = disc[disc["doc_type"].isin(SUBSTANTIVE)]

    frames = []
    for symbol in STOCKS:
        px = pd.read_csv(PRICE_DIR / f"{symbol}.csv", parse_dates=["date"], index_col="date")["Adj Close"]
        ret = clean_returns(px)
        ar = (ret - market.reindex(ret.index)).dropna()
        idx = ar.index

        n_ewm, n_cnt = text_component(news[news["symbol"] == symbol], idx, halflife=10)
        d_ewm, d_cnt = text_component(disc[disc["symbol"] == symbol], idx, halflife=20)
        mom = ar.rolling(60).sum()
        vol = ar.rolling(20).std()

        f = pd.DataFrame({"symbol": symbol, "abn_return": ar, "return": ret.reindex(idx),
                          "news": rolling_z(n_ewm), "disclosure": rolling_z(d_ewm),
                          "momentum": rolling_z(mom), "news_items_30d": n_cnt, "disclosures_30d": d_cnt,
                          "vol_ratio": vol / vol.shift(1).rolling(Z_WINDOW, min_periods=60).median()}, index=idx)
        # Forward abnormal returns from t+1 (what the signal at close t could earn).
        f["fwd5"] = ar[::-1].rolling(5).sum()[::-1].shift(-1)
        f["fwd20"] = ar[::-1].rolling(20).sum()[::-1].shift(-1)
        frames.append(f)

    sig = pd.concat(frames).reset_index(names="date").dropna(subset=["momentum"])
    for name, cols in VARIANTS.items():
        sig[f"score_{name}"] = sig[cols].mean(axis=1, skipna=True)   # average of available components only
        sig[f"signal_{name}"] = np.select([sig[f"score_{name}"] > THRESHOLD, sig[f"score_{name}"] < -THRESHOLD],
                                          ["BUY", "SELL"], "HOLD")
    # Strength of the combined score, reduced for thin text coverage and high volatility.
    coverage = np.clip((sig["news_items_30d"] + sig["disclosures_30d"]) / 5, 0.3, 1.0)
    calm = np.clip(1.5 - 0.5 * sig["vol_ratio"].fillna(1), 0.5, 1.0)
    sig["strength"] = (sig["score_combined"].abs() * coverage * calm).clip(0, 1)
    sig["strength_band"] = pd.cut(sig["strength"], [-0.01, STRENGTH_BANDS[0], STRENGTH_BANDS[1], 1.01],
                                  labels=["Low", "Medium", "High"]).astype(str)
    # Agreement: available components pointing the same way as the combined score.
    comps = sig[VARIANTS["combined"]]
    direction = np.sign(sig["score_combined"])
    sig["n_available"] = comps.notna().sum(axis=1)
    sig["n_agree"] = (np.sign(comps).eq(direction, axis=0) & comps.abs().gt(AGREE_MIN)).sum(axis=1)
    sig["signal"] = sig["signal_combined"]
    return sig


def class_returns(sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in VARIANTS:
        for horizon in ("fwd5", "fwd20"):
            d = sig.dropna(subset=[horizon])
            buy, sell = d.loc[d[f"signal_{name}"] == "BUY", horizon], d.loc[d[f"signal_{name}"] == "SELL", horizon]
            hold = d.loc[d[f"signal_{name}"] == "HOLD", horizon]
            t, p = stats.ttest_ind(buy, sell, equal_var=False)
            rows.append({"variant": name, "horizon": horizon, "n_buy": len(buy), "n_hold": len(hold), "n_sell": len(sell),
                         "BUY_%": buy.mean() * 100, "HOLD_%": hold.mean() * 100, "SELL_%": sell.mean() * 100,
                         "BUY_minus_SELL_%": (buy.mean() - sell.mean()) * 100, "p_value": p,
                         "BUY_hit_rate": (buy > 0).mean(), "SELL_hit_rate": (sell < 0).mean()})
    return pd.DataFrame(rows)


REBALANCE_EVERY = 5  # sessions; the signal is slow-moving, so trade weekly


def positions(score: pd.Series, allow: pd.Series | None = None, every: int = REBALANCE_EVERY) -> pd.Series:
    """Position path with hysteresis: enter at +/-THRESHOLD, exit only when the score crosses 0.

    Without hysteresis the score wobbling around the threshold triggers a trade
    (and a cost) almost every few days, which wiped out the gross return.
    Positions are only revised every `every` sessions; `allow` (optional) restricts
    on which days a new position may be opened.
    """
    allow = np.ones(len(score), bool) if allow is None else allow.to_numpy()
    out, pos = [], 0
    for i, (s, ok) in enumerate(zip(score.to_numpy(), allow)):
        if i % every == 0 and not np.isnan(s):
            if s > THRESHOLD and ok:
                pos = 1
            elif s < -THRESHOLD and ok:
                pos = -1
            elif (pos == 1 and s < 0) or (pos == -1 and s > 0):
                pos = 0
        out.append(pos)
    return pd.Series(out, index=score.index)


def long_short(sig: pd.DataFrame, score_col: str, allow: pd.Series | None = None,
               every: int = REBALANCE_EVERY) -> list[dict]:
    """Equal-weighted long-short across stocks on abnormal returns (market-neutral); gross and net rows."""
    ar = sig.pivot(index="date", columns="symbol", values="abn_return")
    score = sig.pivot(index="date", columns="symbol", values=score_col)
    ok = None if allow is None else \
        sig.assign(allow=allow).pivot(index="date", columns="symbol", values="allow").fillna(False).astype(bool)
    pos = pd.DataFrame({c: positions(score[c], None if ok is None else ok[c], every) for c in score}
                       ).shift(1)                                 # trade the next session
    n_held = pos.abs().sum(axis=1)
    weight = pos.div(n_held.replace(0, np.nan), axis=0)           # equal weight per open position
    gross = (weight * ar).sum(axis=1)
    costs = (weight.fillna(0).diff().abs() * COST).sum(axis=1)
    rows = []
    for label, daily in (("gross", gross), ("net", gross - costs)):
        daily = daily.fillna(0)
        equity = (1 + daily).cumprod()
        years = len(daily) / 250
        rows.append({"returns": label, "days": len(daily),
                     "ann_return_%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
                     "ann_vol_%": daily.std() * np.sqrt(250) * 100,
                     "sharpe": daily.mean() / daily.std() * np.sqrt(250) if daily.std() else np.nan,
                     "max_drawdown_%": ((equity / equity.cummax()) - 1).min() * 100,
                     "time_in_market_%": (n_held > 0).mean() * 100,
                     "trades_per_year": pos.diff().abs().gt(0).sum().sum() / years})
    return rows


def backtest(sig: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"variant": name, **r} for name in VARIANTS for r in long_short(sig, f"score_{name}")])


# Robustness check, added AFTER seeing the main results: does trading less often help?
# Several rules were tried, so treat the best one as optimistic, not as a validated strategy.
def trade_less_rules(sig: pd.DataFrame) -> dict:
    agree = sig["n_agree"].eq(sig["n_available"]) & sig["n_available"].ge(2)
    return {"Weekly, all signals (main result)": {},
            "Weekly, only High-strength signals": {"allow": sig["strength_band"].eq("High")},
            "Weekly, only when all indicators agree": {"allow": agree},
            "Monthly rebalancing": {"every": 20}}


def robustness(sig: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"rule": rule, **r} for rule, kw in trade_less_rules(sig).items()
                         for r in long_short(sig, "score_combined", **kw)])


# Regime analysis. NOTE: this split was chosen after seeing the full-sample results
# (2019-22 = COVID crash/rebound, dominated by reversals; 2023+ = post-COVID, after
# the UAE's 2022 trading-week change). Report it as a regime comparison, not as an
# out-of-sample validation.
PERIODS = {"full": (None, None), "2019-2022": (None, "2022-12-31"), "2023-2026": ("2023-01-01", None)}


def main():
    sig = build_signals()
    sig.to_csv(PROCESSED / "signals.csv", index=False)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    cls_parts, bt_parts = [], []
    for period, (start, end) in PERIODS.items():
        part = sig[(sig["date"] >= (start or "1900")) & (sig["date"] <= (end or "2100"))]
        cls_parts.append(class_returns(part).assign(period=period))
        bt_parts.append(backtest(part).assign(period=period))
    cls, bt = pd.concat(cls_parts), pd.concat(bt_parts)
    cls.to_csv(out / "signal_classes.csv", index=False)
    bt.to_csv(out / "backtest_summary.csv", index=False)
    write_robustness(sig)

    pd.set_option("display.width", 220)
    print(f"{len(sig)} stock-days, {sig['date'].min().date()} -> {sig['date'].max().date()}")
    print("Combined signal mix:\n", sig["signal"].value_counts(normalize=True).round(3).to_string())
    print("\nForward 20-day abnormal return by signal class (%):")
    print(cls[cls["horizon"] == "fwd20"][["period", "variant", "BUY_%", "HOLD_%", "SELL_%", "BUY_minus_SELL_%",
                                          "p_value"]].round(3).to_string(index=False))
    print("\nLong-short backtest on abnormal returns (after 0.2% costs):")
    print(bt[["period", "variant", "returns", "ann_return_%", "ann_vol_%", "sharpe", "max_drawdown_%",
              "trades_per_year"]].round(2).to_string(index=False))
    latest = sig.sort_values("date").groupby("symbol").tail(1)
    print("\nLatest signals:")
    print(latest[["symbol", "date", "news", "disclosure", "momentum", "score_combined", "signal", "strength",
                  "strength_band", "n_agree", "n_available"]].astype({"date": str}).round(2).to_string(index=False))
    print("\nStrength bands (all days):", sig["strength_band"].value_counts(normalize=True).round(2).to_dict())


def write_robustness(sig: pd.DataFrame) -> pd.DataFrame:
    parts = [robustness(sig[(sig["date"] >= (s or "1900")) & (sig["date"] <= (e or "2100"))]).assign(period=p)
             for p, (s, e) in PERIODS.items()]
    rob = pd.concat(parts)
    rob.to_csv(ROOT / "results" / "backtest_robustness.csv", index=False)
    return rob


if __name__ == "__main__":
    import sys
    if "--robustness" in sys.argv:   # only the trade-less check, from the saved signals
        write_robustness(pd.read_csv(PROCESSED / "signals.csv", parse_dates=["date"]))
    else:
        main()
