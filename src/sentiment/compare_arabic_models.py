"""Cross-lingual sentiment: do Arabic NLP approaches agree with English FinBERT
on the SAME disclosure?

DFM publishes each disclosure in English and Arabic. Treating English FinBERT
(our best-validated model) as the reference, we score the Arabic twin with:
  A. CAMeLBERT-MSA sentiment  (Arabic BERT, Modern Standard Arabic)
  B. CAMeLBERT-mix sentiment  (Arabic BERT, MSA + dialects, most widely used)
  C. translate-then-classify  (opus-mt-ar-en machine translation -> FinBERT + domain rule)
No Arabic financial sentiment model exists on Hugging Face, and none of A/B
were trained on financial text, so this measures how well each approach
transfers to formal Arabic financial disclosures - without Arabic labels.

Scope: substantive document types only (press releases, financial
statements, reports), at most MAX_SENTENCES sentences per document, to keep
CPU translation time reasonable.

Output: data/processed/arabic_sentences.csv, data/processed/arabic_docs.csv,
        results/arabic_crosslingual.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from text_processing import clean_arabic
from config import PROCESSED, ROOT, TEXT_AR_DIR
from text_processing.doctype import doc_type
from sentiment.score_news import score_texts
from sentiment.score_documents import load_model

MODELS = Path.home() / "models"
CAMEL = {"CAMeLBERT-MSA": "bert-base-arabic-camelbert-msa-sentiment",
         "CAMeLBERT-mix": "bert-base-arabic-camelbert-mix-sentiment"}
MT_MODEL = "opus-mt-ar-en"
MAX_SENTENCES = 25
SUBSTANTIVE = {"press_release", "financial_statements", "other", "rating", "dividend"}
POS_T, NEG_T = 0.15, -0.15
BATCH = 32


def local(name: str, hub_id: str) -> str:
    p = MODELS / name
    return str(p) if (p / "config.json").exists() else hub_id


@torch.no_grad()
def classify(texts: list[str], path: str) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    probs = []
    for i in tqdm(range(0, len(texts), BATCH), desc=Path(path).name[:30]):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=128, return_tensors="pt")
        probs.append(torch.softmax(model(**enc).logits, dim=-1))
    p = pd.DataFrame(torch.cat(probs).numpy(), columns=labels)
    # CAMeLBERT labels: positive / negative / neutral
    return pd.DataFrame({"polarity": p["positive"] - p["negative"], "label": p.idxmax(axis=1)})


@torch.no_grad()
def translate(texts: list[str]) -> list[str]:
    path = local(MT_MODEL, f"Helsinki-NLP/{MT_MODEL}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).eval()
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: less padding
    out = [None] * len(texts)
    for i in tqdm(range(0, len(texts), BATCH), desc="Translating"):
        idx = order[i:i + BATCH]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=256, return_tensors="pt")
        # Greedy decoding: ~2x faster than beam search on CPU, small quality cost for short formal sentences.
        gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
        for j, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = t
    return out


def english_doc_scores() -> pd.DataFrame:
    """Sentence-weighted English FinBERT score per English disclosure."""
    docs = pd.read_csv(PROCESSED / "documents.csv")[["attachment_id", "disclosure_id", "description"]]
    s = docs.merge(pd.read_csv(PROCESSED / "doc_sentiment.csv"), on="attachment_id").query("n_sentences > 0")
    s["w"] = s["score"] * s["n_sentences"]
    g = s.groupby("disclosure_id").agg(w=("w", "sum"), n=("n_sentences", "sum"), description=("description", "first"))
    return pd.DataFrame({"score_en": g["w"] / g["n"], "doc_type": g["description"].map(doc_type)}).reset_index()


def label(x: pd.Series) -> pd.Series:
    return pd.cut(x, [-1.01, NEG_T, POS_T, 1.01], labels=["negative", "neutral", "positive"]).astype(str)


def main():
    ar_docs = pd.read_csv(PROCESSED / "documents_ar.csv")
    ar_docs = ar_docs[ar_docs["text_path"].notna()]
    en = english_doc_scores()
    pairs = ar_docs.merge(en.rename(columns={"disclosure_id": "disclosure_id_en"}), on="disclosure_id_en")
    pairs = pairs[pairs["doc_type"].isin(SUBSTANTIVE)]

    rows = []
    for d in pairs.itertuples():
        text = (TEXT_AR_DIR / d.text_path.replace("\\", "/")).read_text(encoding="utf-8")
        for i, s in enumerate(clean_arabic.sentences(text, MAX_SENTENCES)):
            rows.append({"attachment_id": d.attachment_id, "disclosure_id_en": d.disclosure_id_en, "sent_idx": i,
                         "sentence_ar": s})
    sents = pd.DataFrame(rows)
    print(f"{len(sents)} Arabic sentences from {sents['attachment_id'].nunique()} documents "
          f"({len(pairs)} paired substantive documents)")

    for name, folder in CAMEL.items():
        r = classify(sents["sentence_ar"].tolist(), local(folder, f"CAMeL-Lab/{folder}"))
        sents[f"{name}_polarity"], sents[f"{name}_label"] = r["polarity"].values, r["label"].values
    sents["translation_en"] = translate(sents["sentence_ar"].tolist())
    tok, model, labels = load_model()
    fb = score_texts(sents["translation_en"], tok, model, labels, "FinBERT on translations")
    sents["MT+FinBERT_polarity"], sents["MT+FinBERT_label"] = fb["polarity"].values, fb["label"].values
    sents.to_csv(PROCESSED / "arabic_sentences.csv", index=False, encoding="utf-8-sig")

    doc = sents.groupby("disclosure_id_en")[[f"{m}_polarity" for m in METHODS]].mean()
    doc = doc.join(en.set_index("disclosure_id")[["score_en", "doc_type"]], how="inner")
    doc.to_csv(PROCESSED / "arabic_docs.csv", encoding="utf-8-sig")
    summarise()


METHODS = ["CAMeLBERT-MSA", "CAMeLBERT-mix", "MT+FinBERT"]


def agreement(doc: pd.DataFrame, group: str) -> list[dict]:
    rows = []
    for m in METHODS:
        x, y = doc[f"{m}_polarity"], doc["score_en"]
        rho, p = stats.spearmanr(x, y)
        rows.append({"doc_type": group, "method": m, "documents": len(doc), "spearman_rho": rho, "spearman_p": p,
                     "pearson_r": stats.pearsonr(x, y)[0], "label_agreement": (label(x) == label(y)).mean(),
                     "cohen_kappa": cohen_kappa_score(label(x), label(y)),
                     "share_positive_ar": (label(x) == "positive").mean(),
                     "share_positive_en": (label(y) == "positive").mean()})
    return rows


def summarise() -> None:
    """Agreement overall AND within each document type.

    Press releases are mostly positive and financial statements mostly
    neutral, so agreement across all documents is partly inflated by the
    document-type split; the within-type figures are the stricter test.
    """
    doc = pd.read_csv(PROCESSED / "arabic_docs.csv")
    rows = agreement(doc, "all")
    for t, g in doc.groupby("doc_type"):
        if len(g) >= 30:
            rows += agreement(g, t)
    res = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "arabic_crosslingual.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAgreement with English FinBERT on the same disclosures:")
    print(res[["doc_type", "method", "documents", "spearman_rho", "label_agreement", "cohen_kappa"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    summarise() if "--summary-only" in sys.argv else main()
