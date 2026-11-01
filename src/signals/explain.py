"""Bilingual, evidence-grounded explanations of a signal with Falcon (RAG).

Pipeline (LangChain):
  1. Look up the stock's signal and its components for the chosen date.
  2. Retrieve that stock's evidence from the Chroma index, filtered to the
     previous LOOKBACK_DAYS, then re-rank by semantic relevance, sentiment
     strength and recency, keeping a mix of sources.
  3. Prompt Falcon-H1 with the signal and the numbered evidence; it must only
     use that evidence and cite it as [n].
  4. Translate the English explanation to Arabic with a second Falcon call.
  5. Faithfulness check: every citation must point to a real evidence item and
     every number in the explanation must appear in the evidence or the signal.

The model is configurable: FALCON_MODEL=tiiuae/Falcon-H1-7B-Instruct on a GPU
server, the 1.5B default on a laptop CPU. Only Falcon-H1 supports Arabic.

Results are cached in data/processed/explanations/<SYMBOL>_<YYYYMMDD>.json
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from signals.build_index import INDEX_DIR, embeddings
from config import PROCESSED, STOCKS

FALCON_MODEL = os.environ.get("FALCON_MODEL", "tiiuae/Falcon-H1-1.5B-Instruct")
# Prefer a copy fetched by download_model.py (~/models/<name>) over the HF cache.
_LOCAL = Path.home() / "models" / FALCON_MODEL.split("/")[-1]
MODEL_PATH = str(_LOCAL) if (_LOCAL / "config.json").exists() and any(_LOCAL.glob("*.safetensors")) else FALCON_MODEL
LOOKBACK_DAYS = 60
N_EVIDENCE = 6
CACHE_DIR = PROCESSED / "explanations"
QUERY = "{company} results profit revenue dividend rating growth decline risk outlook"

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful financial analyst writing for retail investors in the UAE. "
     "Use ONLY the facts in the numbered evidence and the signal summary. "
     "The component scores (news tone, disclosure tone, momentum) come from the model, not from the evidence: "
     "state them without citations. Cite evidence like [1] only for an event or figure that the cited item "
     "itself states. Do not speculate about causes, intentions or the future. "
     "Do not give personal financial advice."),
    ("human",
     "Stock: {company} ({symbol}), Dubai Financial Market. Date: {date}.\n"
     "Model signal: {signal}, signal strength {strength_band} ({n_agree} of {n_available} indicators agree).\n"
     "Components (scale -1 to +1, relative to this stock's own past year): {components}.\n\n"
     "Evidence:\n{evidence}\n\n"
     "In 3 to 4 short sentences, explain why the signal is {signal}: what the recent news and disclosures say "
     "and how the price trend fits. Cite the evidence."),
])
TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional English-to-Arabic financial translator. Translate faithfully into Modern "
               "Standard Arabic. Keep company names, numbers and citation markers like [1] unchanged. "
               "Output only the Arabic translation."),
    ("human", "{text}"),
])


# ------------------------------------------------------------- retrieval ---
@lru_cache(maxsize=1)
def store() -> Chroma:
    return Chroma(collection_name="evidence", embedding_function=embeddings(), persist_directory=str(INDEX_DIR))


def retrieve(symbol: str, asof: pd.Timestamp, k: int = N_EVIDENCE) -> list[dict]:
    start = int((asof - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d"))
    end = int(asof.strftime("%Y%m%d"))
    flt = {"$and": [{"symbol": symbol}, {"date": {"$gte": start}}, {"date": {"$lte": end}}]}
    hits = store().similarity_search_with_relevance_scores(QUERY.format(company=STOCKS[symbol][1]), k=40, filter=flt)
    if not hits:
        return []
    rows = []
    for doc, rel in hits:
        m = doc.metadata
        age = (asof - pd.Timestamp(str(m["date"]))).days
        rows.append({**m, "relevance": rel,
                     # relevance + strength of sentiment + recency (half weight after ~30 days)
                     "rank_score": rel + 0.5 * abs(m["polarity"]) + 0.5 * 0.5 ** (age / 30)})
    df = pd.DataFrame(rows).sort_values("rank_score", ascending=False).drop_duplicates("text")
    # Keep a mix of sources: at most half of the evidence from one kind.
    picked = df.groupby("kind").head(max(2, k // 2)).head(k)
    return picked.sort_values("date", ascending=False).to_dict("records")


def format_evidence(items: list[dict]) -> str:
    return "\n".join(f"[{i}] ({pd.Timestamp(str(e['date'])).date()}, {e['source']}) {e['text']}"
                     for i, e in enumerate(items, 1))


# ------------------------------------------------------------ generation ---
@lru_cache(maxsize=1)
def llm() -> ChatHuggingFace:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=220, do_sample=False,
                   repetition_penalty=1.1, return_full_text=False)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))


def faithfulness(text: str, evidence: list[dict], signal_row: dict, asof: pd.Timestamp) -> dict:
    """Flag citations to non-existent evidence and numbers not found in the sources.

    Limitation: this is a surface check. It cannot tell whether a real
    citation actually supports the claim it is attached to.
    """
    cited = {int(c) for c in re.findall(r"\[(\d+)\]", text)}
    bad_citations = sorted(c for c in cited if not 1 <= c <= len(evidence))
    shown = " ".join(f"{signal_row[k]:+.2f} {signal_row[k]:.2f} {abs(signal_row[k]):.2f}"
                     for k in ("news", "disclosure", "momentum", "strength")
                     if signal_row[k] is not None)  # values as displayed
    source_text = (" ".join(e["text"] for e in evidence) + " " + json.dumps(signal_row) + " " + shown +
                   f" {asof:%d %B %Y} {asof:%Y-%m-%d} {asof.day}")
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"\[\d+\]", "", text)))
    unsupported = sorted(n for n in numbers if n not in source_text and len(n) > 1)
    return {"citations": sorted(cited), "bad_citations": bad_citations, "unsupported_numbers": unsupported,
            "grounded": not bad_citations and not unsupported and bool(cited)}


ARABIC_MIN_SHARE = 0.8


def arabic_quality(text: str) -> tuple[bool, str]:
    """Accept a translation only if it is mostly Arabic script and has no CJK characters.

    Returns (accepted, reason) where reason describes why it was rejected.
    """
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[぀-ヿ一-鿿가-힯]", text))
    share = arabic / max(arabic + latin + cjk, 1)
    reasons = ([f"only {share:.0%} Arabic script"] if share < ARABIC_MIN_SHARE else []) + \
              ([f"{cjk} Chinese/Japanese/Korean characters"] if cjk else [])
    return not reasons, ", ".join(reasons)


def model_available() -> bool:
    """True if the Falcon weights are on disk (local folder or HF cache)."""
    if Path(MODEL_PATH).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(FALCON_MODEL, "model.safetensors"), str)
    except Exception:
        return False


COMPONENT_NAMES = {"news": ("news tone", "نبرة الأخبار"),
                   "disclosure": ("company disclosure tone", "نبرة إفصاحات الشركة"),
                   "momentum": ("price momentum versus the market", "زخم السعر مقارنة بالسوق")}
BAND_AR = {"Low": "منخفضة", "Medium": "متوسطة", "High": "مرتفعة"}


def signal_summary(row) -> dict:
    """Signal fields used in prompts, templates and checks; missing components become None."""
    def num(v):
        return None if pd.isna(v) else round(float(v), 2)
    return {"signal": row["signal"], "strength": num(row["strength"]), "strength_band": row["strength_band"],
            "n_agree": int(row["n_agree"]), "n_available": int(row["n_available"]),
            "news": num(row["news"]), "disclosure": num(row["disclosure"]), "momentum": num(row["momentum"])}


def components_text(row: dict) -> str:
    return ", ".join(f"{name} {'not available (no recent items)' if row[k] is None else f'{row[k]:+.2f}'}"
                     for k, (name, _) in COMPONENT_NAMES.items())


def template_explanation(row: dict, evidence: list[dict]) -> tuple[str, str]:
    """Deterministic fallback when no LLM is available: same evidence, fixed wording.

    Components are relative to each stock's own past year, so they are worded
    as "above / below its usual level" rather than "positive / negative".
    """
    def tone(v):
        # Arabic uses gender-neutral phrasing ("than the usual level") because the
        # component nouns differ in grammatical gender (نبرة is feminine, زخم masculine).
        return ("well above its usual level", "أعلى بكثير من المستوى المعتاد") if v > 0.5 else \
               ("above its usual level", "أعلى من المستوى المعتاد") if v > 0.15 else \
               ("well below its usual level", "أدنى بكثير من المستوى المعتاد") if v < -0.5 else \
               ("below its usual level", "أدنى من المستوى المعتاد") if v < -0.15 else \
               ("close to its usual level", "في حدود المستوى المعتاد")
    en = [f"Signal: {row['signal']}, strength {row['strength_band']} "
          f"({row['n_agree']} of {row['n_available']} indicators agree)."]
    ar = [f"الإشارة: {row['signal']}، قوة الإشارة {BAND_AR.get(row['strength_band'], row['strength_band'])} "
          f"({row['n_agree']} من {row['n_available']} مؤشرات متوافقة)."]
    for k, (name_en, name_ar) in COMPONENT_NAMES.items():
        if row[k] is None:
            en.append(f"No recent items for {name_en}, so it is not used.")
            ar.append(f"لا توجد بيانات حديثة عن {name_ar}، لذلك لم تُستخدم.")
            continue
        t_en, t_ar = tone(row[k])
        en.append(f"Recent {name_en} is {t_en} ({row[k]:+.2f}).")
        ar.append(f"{name_ar} {t_ar} ({row[k]:+.2f}).")
    if evidence:
        pos = max(range(len(evidence)), key=lambda i: evidence[i]["polarity"])
        neg = min(range(len(evidence)), key=lambda i: evidence[i]["polarity"])
        if evidence[pos]["polarity"] > 0.3:
            en.append(f"The most positive recent item: \"{evidence[pos]['text'][:160]}\" [{pos + 1}].")
            ar.append(f"أكثر البنود إيجابية مؤخراً مذكور في المصدر [{pos + 1}].")
        if evidence[neg]["polarity"] < -0.3:
            en.append(f"The most negative recent item: \"{evidence[neg]['text'][:160]}\" [{neg + 1}].")
            ar.append(f"أكثر البنود سلبية مؤخراً مذكور في المصدر [{neg + 1}].")
    return " ".join(en), " ".join(ar)


def has_llm_cache(symbol: str, date) -> bool:
    """True if a Falcon explanation for this stock and session is already cached."""
    return (CACHE_DIR / f"{symbol}_{pd.Timestamp(date):%Y%m%d}.json").exists()


def explain(symbol: str, date: str | None = None, use_cache: bool = True, use_llm: bool | None = None) -> dict:
    sig = pd.read_csv(PROCESSED / "signals.csv", parse_dates=["date"])
    sig = sig[sig["symbol"] == symbol]
    row = sig[sig["date"] <= pd.Timestamp(date)].iloc[-1] if date else sig.iloc[-1]
    asof = row["date"]
    use_llm = model_available() if use_llm is None else use_llm
    cache = CACHE_DIR / f"{symbol}_{asof:%Y%m%d}{'' if use_llm else '_template'}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    evidence = retrieve(symbol, asof)
    signal_row = signal_summary(row)
    if use_llm:
        inputs = {"company": STOCKS[symbol][1], "symbol": symbol, "date": f"{asof:%d %B %Y}",
                  **signal_row, "components": components_text(signal_row),
                  "evidence": format_evidence(evidence) or "(no recent news or disclosures)"}
        english = (EXPLAIN_PROMPT | llm() | StrOutputParser()).invoke(inputs).strip()
        arabic = (TRANSLATE_PROMPT | llm() | StrOutputParser()).invoke({"text": english}).strip()
        model_name = FALCON_MODEL
        arabic_ok, reason = arabic_quality(arabic)
        if not arabic_ok:
            # Small models drift into other languages; never show a broken translation.
            arabic = template_explanation(signal_row, evidence)[1]
            arabic_source = f"template (Falcon translation rejected: {reason})"
        else:
            arabic_source = FALCON_MODEL
    else:
        english, arabic = template_explanation(signal_row, evidence)
        model_name = arabic_source = "template (no LLM)"

    result = {"symbol": symbol, "date": f"{asof:%Y-%m-%d}", "model": model_name, "arabic_source": arabic_source,
              **signal_row, "explanation_en": english, "explanation_ar": arabic, "evidence": evidence,
              "check": faithfulness(english, evidence, signal_row, asof)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for sym in (sys.argv[1:] or ["DEWA"]):
        r = explain(sym, use_cache=False)
        print(f"\n===== {sym} {r['date']}: {r['signal']} (strength {r['strength_band']}, "
              f"{r['n_agree']}/{r['n_available']} agree) =====")
        print(format_evidence(r["evidence"]))
        print("\nEN:", r["explanation_en"])
        print("\nAR:", r["explanation_ar"])
        print("\nCheck:", r["check"])
