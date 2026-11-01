"""Efsah Insight - Streamlit demo.

Run from the project root:  streamlit run src/app/app.py

Pick an exchange and a stock to see:
  * the Buy / Hold / Sell signal, its strength (Low / Medium / High), indicator agreement and components
  * the price chart with company disclosures and news headlines marked by sentiment
  * a bilingual (English / Arabic) explanation grounded in cited evidence
    (Falcon-H1 via LangChain RAG, or a deterministic template if no LLM is available)
  * the evaluation results behind the model
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PRICE_DIR, PROCESSED, ROOT, STOCKS  # noqa: E402

# Reference palette (dataviz skill): diverging blue <-> red with a gray midpoint.
POS, NEG, NEUTRAL, LINE = "#2a78d6", "#e34948", "#898781", "#2a78d6"
SIGNAL_ICON = {"BUY": "▲ BUY", "HOLD": "● HOLD", "SELL": "▼ SELL"}
STRENGTH_ICON = {"Low": "▁", "Medium": "▃", "High": "▆"}
EXCHANGES = {"Dubai Financial Market (DFM)": list(STOCKS)}

st.set_page_config(page_title="Efsah Insight", page_icon="📊", layout="wide")


# ----------------------------------------------------------------- data ---
# Each loader takes the source file's modification time as an argument, so
# Streamlit's cache is invalidated automatically whenever the pipeline rewrites
# a file (otherwise a running app keeps serving stale data with old columns).
def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _load_signals(version: float) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "signals.csv", parse_dates=["date"])


def load_signals() -> pd.DataFrame:
    return _load_signals(mtime(PROCESSED / "signals.csv"))


@st.cache_data
def _load_prices(symbol: str, version: float) -> pd.Series:
    return pd.read_csv(PRICE_DIR / f"{symbol}.csv", parse_dates=["date"], index_col="date")["Close"]


def load_prices(symbol: str) -> pd.Series:
    return _load_prices(symbol, mtime(PRICE_DIR / f"{symbol}.csv"))


def load_events(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    version = mtime(PROCESSED / "events_disclosures.csv") + mtime(PROCESSED / "news_sentiment.csv")
    return _load_events(symbol, version)


@st.cache_data
def _load_events(symbol: str, version: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    disc = pd.read_csv(PROCESSED / "events_disclosures.csv", parse_dates=["day0", "published"])
    heads = pd.read_csv(PROCESSED.parent / "raw" / "disclosures.csv")[["id", "headline_en", "headline_ar"]]
    disc = disc.merge(heads, left_on="disclosure_id", right_on="id", how="left")
    news = pd.read_csv(PROCESSED / "news_sentiment.csv")
    news = news[(news["symbol"] == symbol) & news["names_company"]].copy()
    news["published"] = pd.to_datetime(news["published"], utc=True).dt.tz_convert("Asia/Dubai").dt.tz_localize(None)
    return disc[disc["symbol"] == symbol], news


@st.cache_data
def _load_results(name: str, version: float) -> pd.DataFrame | None:
    p = ROOT / "results" / name
    return pd.read_csv(p) if p.exists() else None


def load_results(name: str) -> pd.DataFrame | None:
    return _load_results(name, mtime(ROOT / "results" / name))


def tone_color(v: float) -> str:
    return POS if v > 0.15 else NEG if v < -0.15 else NEUTRAL


# --------------------------------------------------------------- charts ---
def price_chart(prices: pd.Series, disc: pd.DataFrame, news: pd.DataFrame, start: pd.Timestamp) -> go.Figure:
    p = prices[prices.index >= start]
    fig = go.Figure(go.Scatter(x=p.index, y=p.values, mode="lines", name="Close price (AED)",
                               line=dict(color=LINE, width=2),
                               hovertemplate="%{x|%d %b %Y}<br>Close AED %{y:.2f}<extra></extra>"))
    for df, name, symbol_shape, text_col, date_col, score_col in (
            (disc, "Company disclosure", "diamond", "headline_en", "published", "score"),
            (news, "News headline", "circle", "headline", "published", "polarity")):
        d = df[df[date_col] >= start].copy()
        if d.empty:
            continue
        d["y"] = p.reindex(d[date_col].dt.normalize(), method="ffill").values
        fig.add_trace(go.Scatter(
            x=d[date_col], y=d["y"], mode="markers", name=name,
            marker=dict(symbol=symbol_shape, size=10, color=[tone_color(v) for v in d[score_col]],
                        line=dict(width=2, color="white")),
            customdata=list(zip(d[text_col].fillna("").str.slice(0, 90), d[score_col])),
            hovertemplate="%{x|%d %b %Y}<br>%{customdata[0]}<br>sentiment %{customdata[1]:+.2f}<extra>" + name + "</extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title="AED")
    fig.update_xaxes(showgrid=False)
    return fig


def components_chart(row: pd.Series) -> go.Figure:
    names = ["News tone", "Disclosure tone", "Price momentum"]
    raw = [row["news"], row["disclosure"], row["momentum"]]
    vals = [0 if pd.isna(v) else v for v in raw]
    labels = ["no recent items (not used)" if pd.isna(v) else f"{v:+.2f}" for v in raw]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[tone_color(v) for v in vals],
                           text=labels, textposition="outside", customdata=labels,
                           hovertemplate="%{y}: %{customdata}<extra></extra>"))
    fig.update_layout(height=200, margin=dict(l=10, r=40, t=10, b=10), showlegend=False,
                      xaxis=dict(range=[-1.15, 1.15], zeroline=True, zerolinecolor=NEUTRAL, title="below usual  ←  0  →  above usual"))
    fig.update_yaxes(autorange="reversed")
    return fig


def score_history_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=hist["date"], y=hist["score_combined"], mode="lines", name="Combined score",
                               line=dict(color=LINE, width=2),
                               hovertemplate="%{x|%d %b %Y}<br>score %{y:+.2f}<extra></extra>"))
    for y, label in ((0.25, "BUY threshold"), (-0.25, "SELL threshold")):
        fig.add_hline(y=y, line=dict(color=NEUTRAL, dash="dot", width=1), annotation_text=label,
                      annotation_position="top left")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[-1, 1], title="score"))
    fig.update_xaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------- UI ---
st.title("📊 Efsah Insight")
st.caption("Bilingual NLP decision support for Dubai Financial Market stocks: official company disclosures, "
           "news sentiment (FinBERT) and price trends, explained with Falcon (TII, Abu Dhabi).")
st.warning("**Decision support only, not financial advice.** Signals are experimental; in back-tests the "
           "after-cost edge was small and unstable over time. Always read the cited sources.", icon="⚠️")

with st.sidebar:
    st.header("Select")
    exchange = st.selectbox("Exchange", list(EXCHANGES))
    symbol = st.selectbox("Stock", EXCHANGES[exchange], format_func=lambda s: f"{STOCKS[s][1]} ({s})")
    signals = load_signals()
    hist = signals[signals["symbol"] == symbol].sort_values("date")
    asof = st.date_input("As of", value=hist["date"].max().date(), min_value=hist["date"].min().date(),
                         max_value=hist["date"].max().date())
    lookback = st.select_slider("Chart window", ["3M", "6M", "1Y", "3Y"], value="6M")
    st.divider()
    st.caption("Sources: DFM disclosures, Google News, Mubasher, Emirates NBD call transcripts, Yahoo Finance prices.")

row = hist[hist["date"] <= pd.Timestamp(asof)].iloc[-1]
company, sector = STOCKS[symbol][1], STOCKS[symbol][2]

# Headline: signal tile + components
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader(f"{company}")
    st.caption(f"{symbol} · {sector} · signal as of {row['date']:%d %b %Y}")
    st.metric("Signal", SIGNAL_ICON[row["signal"]])
    band = row["strength_band"]
    st.markdown(f"**Signal strength: {STRENGTH_ICON[band]} {band}** · "
                f"{int(row['n_agree'])} of {int(row['n_available'])} indicators agree")
    st.caption("Strength is how clear-cut the inputs are, not a probability of being right "
               "(in back-tests BUY signals were right about half the time).")
    st.caption(f"Text coverage (last 30 sessions): {int(row['news_items_30d'])} news days, "
               f"{int(row['disclosures_30d'])} disclosures")
with c2:
    st.markdown("**What drives the signal** (each relative to this stock's own past year)")
    st.plotly_chart(components_chart(row), width="stretch")

# Price chart with events
start = row["date"] - pd.DateOffset(months={"3M": 3, "6M": 6, "1Y": 12, "3Y": 36}[lookback])
disc, news = load_events(symbol)
disc, news = disc[disc["published"] <= row["date"] + pd.Timedelta(days=1)], news[news["published"] <= row["date"] + pd.Timedelta(days=1)]
st.markdown("**Price with disclosures (◆) and news (●)**: blue = positive, red = negative, gray = neutral")
st.plotly_chart(price_chart(load_prices(symbol), disc, news, start), width="stretch")

# Explanation
st.markdown("### Why? (English / العربية)")
try:
    from signals import explain
    llm_ready = explain.model_available()
except Exception as e:  # missing optional dependencies
    explain, llm_ready = None, False
    st.info(f"Explanation module unavailable: {e}")

if explain:
    cached = explain.has_llm_cache(symbol, row["date"])
    use_llm = st.toggle(f"Use Falcon ({explain.FALCON_MODEL.split('/')[-1]})", value=cached,
                        disabled=not (llm_ready or cached),
                        help="Cached Falcon explanations load instantly; generating a new one on a laptop CPU "
                             "takes several minutes. Off = deterministic template from the same evidence.")
    with st.spinner("Retrieving evidence and writing the explanation…"):
        result = explain.explain(symbol, str(row["date"].date()), use_llm=use_llm)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**English**")
        st.write(result["explanation_en"])
    with e2:
        st.markdown("**العربية**")
        st.markdown(f"<div dir='rtl' style='text-align:right'>{result['explanation_ar']}</div>", unsafe_allow_html=True)
    check = result["check"]
    status = "✅ Grounded: every citation and number traced to the evidence" if check["grounded"] else \
        f"⚠️ Check: bad citations {check['bad_citations']}, unverified numbers {check['unsupported_numbers']}"
    st.caption(f"English: {result['model']} · Arabic: {result.get('arabic_source', result['model'])} · {status}")

    st.markdown("**Evidence used** (retrieved from the last 60 days)")
    ev = pd.DataFrame(result["evidence"])
    if ev.empty:
        st.info("No recent news or disclosures for this stock.")
    else:
        ev.insert(0, "#", range(1, len(ev) + 1))
        ev["date"] = pd.to_datetime(ev["date"].astype(str)).dt.date
        st.dataframe(ev[["#", "date", "kind", "source", "text", "polarity", "url"]], hide_index=True,
                     width="stretch",
                     column_config={"url": st.column_config.LinkColumn("link"),
                                    "polarity": st.column_config.NumberColumn("sentiment", format="%+.2f")})

# Tabs with history and evaluation
t1, t2, t3, t4 = st.tabs(["Signal history", "Model evaluation", "Does sentiment move prices?", "About & ethics"])
with t1:
    st.plotly_chart(score_history_chart(hist[hist["date"] >= start]), width="stretch")
    bt = load_results("backtest_summary.csv")
    if bt is not None:
        st.markdown("**Back-test** (market-neutral long-short on abnormal returns, weekly rebalancing)")
        st.dataframe(bt[["period", "variant", "returns", "ann_return_%", "sharpe", "max_drawdown_%"]].round(2),
                     hide_index=True, width="stretch")
    tl = load_results("backtest_robustness.csv")
    if tl is not None:
        st.markdown("**Robustness check: does trading less often help?** (combined signal, after 0.2% costs)")
        st.dataframe(tl[tl["returns"] == "net"][["period", "rule", "ann_return_%", "sharpe", "max_drawdown_%",
                                                  "trades_per_year"]].round(2), hide_index=True, width="stretch")
        st.caption("Tested after seeing the main results and several rules were tried, so the best row is "
                   "optimistic. Trading less cuts costs and helps after 2023, but no rule is profitable over the "
                   "full period; it would need confirming on new data.")
with t2:
    ev_res = load_results("model_eval.csv")
    if ev_res is not None:
        st.markdown("**Sentiment models on labelled benchmarks** (FinBERT was trained on Financial PhraseBank, "
                    "so the Twitter set is the fairer test)")
        st.dataframe(ev_res[["dataset", "model", "n", "accuracy", "macro_f1"]].round(3), hide_index=True,
                     width="stretch")
with t3:
    es = load_results("event_study_summary.csv")
    if es is not None:
        st.markdown("**Event study**: abnormal returns after events, most positive vs most negative third")
        cols = ["source", "signal", "window", "n", "spearman_rho", "spearman_p", "top_minus_bottom_%", "t_test_p"]
        st.dataframe(es[es["window"].isin(["CAR[0,1]", "CAR[0,4]"])][cols].round(3), hide_index=True,
                     width="stretch")
with t4:
    st.markdown("""
**How it works.** DFM disclosures (PDF → text, OCR for scans) and news headlines are scored with **FinBERT**
plus a domain rule for "decrease of a cost = good news". Scores are compared with each stock's own history
(companies always write positively), combined with price momentum, and turned into a transparent
Buy / Hold / Sell rule. **Falcon-H1** (TII, Abu Dhabi) writes a bilingual explanation using **only** evidence
retrieved from a Chroma vector index via LangChain, and every citation and number is checked automatically.

**Responsible AI.**
- Not financial advice: signal strength shown as Low / Medium / High with indicator agreement - deliberately
  not a "confidence %", which would imply a probability the back-tests do not support. Sources linked,
  back-test limitations stated.
- Missing information is treated as missing (excluded), never as neutral.
- No look-ahead: only information published before each trading session is used.
- Hallucination control: retrieval-grounded prompts + automatic citation / number checks + template fallback.
- Entity errors handled: e.g. "Dewa United" (football club) and "Emaar The Economic City" are filtered out.
- Data: public DFM disclosures, headlines and links only (no full articles stored); polite rate limits.
- Arabic analysis is limited to output; Arabic sentiment and Gulf-dialect bias testing are future work.
""")
