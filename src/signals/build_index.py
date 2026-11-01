"""Build the evidence index for retrieval-augmented explanations.

Every piece of evidence is a short, dated, citable text with metadata:
  * news        - company-named Google News headlines
  * disclosure  - the 3 most strongly worded sentences of each DFM disclosure, prefixed by its headline
  * mubasher    - English Mubasher headline + lead paragraph
  * call        - strongest management / analyst sentences from Emirates NBD results calls

Texts are embedded with multilingual-e5-small (supports Arabic, so Arabic
sources can be added later without changing the model) and stored in a
persistent Chroma vector store via LangChain. Metadata (symbol, date as an
int YYYYMMDD, kind, polarity, url) allows filtered retrieval such as
"EMAAR, last 60 days".

Output: ~/.cache/dfm_nlp/index (Chroma database; set DFM_INDEX_DIR to change it), data/processed/evidence.csv (the same texts as a table)
"""
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import DOCS_BASE, PROCESSED, RAW, STOCKS

# Chroma keeps a SQLite database; cloud-sync clients such as OneDrive lock its
# files while syncing, which breaks rebuilds and can corrupt it. The index is a
# derived cache (rebuilt in ~2 minutes), so it lives outside the project folder.
INDEX_DIR = Path(os.environ.get("DFM_INDEX_DIR", Path.home() / ".cache" / "dfm_nlp" / "index"))
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_SENTENCES = 3       # per disclosure
TOP_CALL_SENTENCES = 8  # per call and role


def embeddings() -> HuggingFaceEmbeddings:
    # e5 models expect "query: " / "passage: " prefixes.
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                 encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
                                 query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True})


def as_int_date(ts) -> int:
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


def news_evidence() -> pd.DataFrame:
    n = pd.read_csv(PROCESSED / "news_sentiment.csv")
    n = n[n["names_company"]]
    published = pd.to_datetime(n["published"], utc=True).dt.tz_convert("Asia/Dubai")
    return pd.DataFrame({"kind": "news", "symbol": n["symbol"], "date": published.map(as_int_date),
                         "source": n["source"], "text": n["headline"], "polarity": n["polarity"], "url": n["link"]})


def disclosure_evidence() -> pd.DataFrame:
    docs = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["published"])
    heads = pd.read_csv(RAW / "disclosures.csv")[["id", "headline_en"]].rename(columns={"id": "disclosure_id"})
    sents = pd.read_csv(PROCESSED / "sentences.csv")
    s = sents.merge(docs[["attachment_id", "disclosure_id", "symbol", "published", "r_path"]], on="attachment_id")
    s = s.merge(heads, on="disclosure_id", how="left")
    s["strength"] = s["polarity"].abs()
    top = s.sort_values("strength", ascending=False).groupby("attachment_id").head(TOP_SENTENCES)
    return pd.DataFrame({"kind": "disclosure", "symbol": top["symbol"], "date": top["published"].map(as_int_date),
                         "source": "DFM disclosure: " + top["headline_en"].fillna(""),
                         "text": top["sentence"], "polarity": top["polarity"],
                         "url": DOCS_BASE + top["r_path"].map(quote)})


def mubasher_evidence() -> pd.DataFrame:
    m = pd.read_csv(PROCESSED / "mubasher_sentiment.csv", parse_dates=["published"])
    text = m["headline"] + ". " + m["lead"].fillna("").str.replace(r"\s*…$", "", regex=True)
    return pd.DataFrame({"kind": "mubasher", "symbol": m["symbol"], "date": m["published"].map(as_int_date),
                         "source": "Mubasher", "text": text.str.strip(" ."), "polarity": m["headline_polarity"],
                         "url": m["link"]})


def call_evidence() -> pd.DataFrame:
    t = pd.read_csv(PROCESSED / "transcript_sentences.csv").dropna(subset=["call_date"])
    t["strength"] = t["polarity"].abs()
    top = (t[t["role"].isin(["management", "analyst"])].sort_values("strength", ascending=False)
           .groupby(["period", "role"]).head(TOP_CALL_SENTENCES))
    who = top["role"].map({"management": "ENBD management", "analyst": "Analyst"})
    return pd.DataFrame({"kind": "call", "symbol": "EMIRATESNBD", "date": pd.to_datetime(top["call_date"]).map(as_int_date),
                         "source": who + " on " + top["period"] + " results call (" + top["speaker"] + ")",
                         "text": top["sentence"], "polarity": top["polarity"],
                         "url": "https://www.emiratesnbd.com/en/investor-relations"})


def main():
    ev = pd.concat([news_evidence(), disclosure_evidence(), mubasher_evidence(), call_evidence()], ignore_index=True)
    ev = ev[ev["symbol"].isin(STOCKS)].dropna(subset=["text"]).drop_duplicates(["symbol", "text"])
    ev["company"] = ev["symbol"].map(lambda s: STOCKS[s][1])
    ev.to_csv(PROCESSED / "evidence.csv", index=False, encoding="utf-8-sig")
    print(ev.groupby(["kind"]).size().to_string(), f"\ntotal {len(ev)}")

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    docs = [Document(page_content=f"{r.company}: {r.text}",
                     metadata={"kind": r.kind, "symbol": r.symbol, "date": int(r.date), "source": str(r.source),
                               "polarity": float(r.polarity), "url": str(r.url), "text": r.text})
            for r in ev.itertuples()]
    store = Chroma(collection_name="evidence", embedding_function=embeddings(), persist_directory=str(INDEX_DIR))
    batch = 2000
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i:i + batch])
        print(f"  indexed {min(i + batch, len(docs))}/{len(docs)}", flush=True)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
