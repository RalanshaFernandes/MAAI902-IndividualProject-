"""Score every readable document with FinBERT.

FinBERT (ProsusAI/finbert) classifies a sentence as positive, negative or
neutral. Documents are scored sentence by sentence and then aggregated, which
avoids FinBERT's 512-token limit and keeps track of *which* sentences drove
the result (useful for explanations later).

Document score = mean(P(positive) - P(negative)) over its sentences, in [-1, 1].

Output: data/processed/sentences.csv      one row per scored sentence
        data/processed/doc_sentiment.csv  one row per document
"""
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment import domain_rules
from config import PROCESSED, TEXT_DIR
from text_processing.clean_english import sentences

MODEL = "ProsusAI/finbert"
MAX_SENTENCES = 150   # long financial statements: the narrative summary comes first
BATCH = 32
# Thresholds on the document score for a 3-way label.
POS_T, NEG_T = 0.15, -0.15


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    return tok, model, labels


@torch.no_grad()
def score(texts: list[str], tok, model, labels) -> pd.DataFrame:
    """Return P(positive), P(negative), P(neutral) for each sentence."""
    probs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=128, return_tensors="pt")
        probs.append(torch.softmax(model(**enc).logits, dim=-1))
    p = torch.cat(probs).numpy()
    return pd.DataFrame(p, columns=[f"p_{l}" for l in labels])


def main(rescore_all: bool = False):
    docs = pd.read_csv(PROCESSED / "documents.csv")
    docs = docs[~docs["scanned"].fillna(True).astype(bool)]
    readable = set(docs["attachment_id"])
    # Incremental: reuse FinBERT scores for documents scored in an earlier run
    # (domain_rules.apply undoes and redoes their flips in aggregate()).
    done = pd.DataFrame()
    if (PROCESSED / "sentences.csv").exists() and not rescore_all:
        done = pd.read_csv(PROCESSED / "sentences.csv")
        done = done[done["attachment_id"].isin(readable)]
        docs = docs[~docs["attachment_id"].isin(done["attachment_id"])]
    print(f"{len(docs)} documents to score, {done['attachment_id'].nunique() if len(done) else 0} reused")
    tok, model, labels = load_model()
    torch.set_num_threads(max(torch.get_num_threads(), 4))

    sent_rows, empty = [], []
    for d in tqdm(docs.itertuples(), total=len(docs), desc="FinBERT"):
        text = (TEXT_DIR / d.text_path.replace("\\", "/")).read_text(encoding="utf-8")
        sents = sentences(text, MAX_SENTENCES)
        if not sents:
            empty.append(d.attachment_id)
            continue
        s = score(sents, tok, model, labels)
        s.insert(0, "attachment_id", d.attachment_id)
        s.insert(1, "sent_idx", range(len(sents)))
        s.insert(2, "sentence", sents)
        sent_rows.append(s)

    sents = pd.concat([done, *sent_rows], ignore_index=True)
    sents.to_csv(PROCESSED / "sentences.csv", index=False, encoding="utf-8-sig")
    aggregate(sents, sorted(readable - set(sents["attachment_id"])))


def aggregate(sents: pd.DataFrame, empty: list[str] = ()) -> None:
    """Apply the domain rule to sentence scores and aggregate them per document.

    Kept separate from scoring so document scores can be rebuilt from
    sentences.csv (python -m sentiment.score_documents --reaggregate) without rerunning FinBERT.
    """
    sents = domain_rules.apply(sents, "sentence")
    sents.to_csv(PROCESSED / "sentences.csv", index=False, encoding="utf-8-sig")
    doc = sents.groupby("attachment_id").agg(
        n_sentences=("polarity", "size"), score=("polarity", "mean"),
        share_positive=("label", lambda x: (x == "positive").mean()),
        share_negative=("label", lambda x: (x == "negative").mean()),
        share_neutral=("label", lambda x: (x == "neutral").mean())).reset_index()
    doc = pd.concat([doc, pd.DataFrame({"attachment_id": list(empty), "n_sentences": 0})], ignore_index=True)
    doc["label"] = pd.cut(doc["score"], [-1.01, NEG_T, POS_T, 1.01], labels=["negative", "neutral", "positive"])
    doc.to_csv(PROCESSED / "doc_sentiment.csv", index=False)

    scored = doc[doc["n_sentences"] > 0]
    print(f"\nScored {len(scored)} documents ({scored['n_sentences'].sum()} sentences); "
          f"{(doc['n_sentences'] == 0).sum()} had no usable sentences; "
          f"domain rule flipped {int(sents['rule_flipped'].sum())} sentences")
    print(scored["label"].value_counts().to_string())


def reaggregate() -> None:
    sents = pd.read_csv(PROCESSED / "sentences.csv")
    docs = pd.read_csv(PROCESSED / "documents.csv")
    empty = sorted(set(docs["attachment_id"]) - set(sents["attachment_id"]))
    aggregate(sents, empty)  # domain_rules.apply undoes earlier flips, so this is safe to repeat


if __name__ == "__main__":
    import sys
    reaggregate() if "--reaggregate" in sys.argv else main(rescore_all="--all" in sys.argv)
