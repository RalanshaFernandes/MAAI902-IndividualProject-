"""Build the sample for the hand-labelled DFM evaluation set ("gold set").

FinBERT has only been evaluated on US/European benchmarks. This sample lets
us measure it on UAE text: 150 DFM disclosure sentences + 100 news headlines.

Sampling design:
  * Stratified by FinBERT's predicted label (positive / neutral / negative),
    because negatives are rare (~3% of disclosure sentences) and a simple
    random sample would contain almost none. Evaluation re-weights by each
    stratum's share of the full data to estimate population accuracy.
  * Includes sentences changed by the domain rule, to test it on DFM text.
  * At most one sentence per document, 6-60 words, so items are varied.
  * Shuffled with a fixed seed; the labeller never sees model predictions.

Output: labels/gold_sample.csv (items + hidden model predictions)
        labels/strata.csv      (population share of each stratum, for re-weighting)
"""
import sys

import pandas as pd

from config import PROCESSED, ROOT, STOCKS

SEED = 902
LABEL_DIR = ROOT / "labels"
PER_STRATUM_DISC = 45   # x3 labels = 135, + flipped sentences = ~150
N_FLIPPED = 15
PER_STRATUM_NEWS = 34   # x3 = ~100


def disclosure_items() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(PROCESSED / "sentences.csv")
    docs = pd.read_csv(PROCESSED / "documents.csv")[["attachment_id", "symbol", "description"]]
    s = s.merge(docs, on="attachment_id")
    words = s["sentence"].str.split().str.len()
    pool = s[(words >= 6) & (words <= 60)]
    strata = pool["label"].value_counts(normalize=True).rename("population_share")

    flipped = pool[pool["rule_flipped"]].sample(min(N_FLIPPED, int(pool["rule_flipped"].sum())), random_state=SEED)
    rest = pool[~pool["attachment_id"].isin(flipped["attachment_id"])].sample(frac=1, random_state=SEED)
    # One sentence per document for variety; negatives are concentrated in few
    # documents (e.g. 2020 results), so allow up to 3 per document for them.
    per_doc = {"positive": 1, "neutral": 1, "negative": 3}
    picked = pd.concat([rest[rest["label"] == lab].groupby("attachment_id").head(per_doc[lab])
                        .drop_duplicates("sentence").head(PER_STRATUM_DISC) for lab in per_doc])
    items = pd.concat([picked.assign(stratum=picked["label"]), flipped.assign(stratum="rule_flipped")])
    out = pd.DataFrame({"source": "DFM disclosure", "symbol": items["symbol"], "text": items["sentence"],
                        "context": items["description"], "stratum": items["stratum"],
                        "finbert_raw": items[["p_positive", "p_negative", "p_neutral"]].idxmax(axis=1).str[2:]
                        .where(~items["rule_flipped"], "negative"),
                        "finbert_rule": items["label"]})
    return out, strata.rename_axis("label").reset_index().assign(source="DFM disclosure")


def news_items() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = pd.read_csv(PROCESSED / "news_sentiment.csv")
    n = n[n["names_company"]].drop_duplicates("headline")
    strata = n["label"].value_counts(normalize=True).rename("population_share")
    picked = pd.concat([n[n["label"] == lab].sample(PER_STRATUM_NEWS, random_state=SEED)
                        for lab in ("positive", "neutral", "negative")])
    raw = picked[["p_positive", "p_negative", "p_neutral"]].copy()
    raw.loc[picked["rule_flipped"], ["p_positive", "p_negative"]] = \
        raw.loc[picked["rule_flipped"], ["p_negative", "p_positive"]].to_numpy()   # undo the rule
    out = pd.DataFrame({"source": "News headline", "symbol": picked["symbol"], "text": picked["headline"],
                        "context": picked["source"], "stratum": picked["label"],
                        "finbert_raw": raw.idxmax(axis=1).str[2:], "finbert_rule": picked["label"]})
    return out, strata.rename_axis("label").reset_index().assign(source="News headline")


def main():
    # The sample was drawn once (October 2026) and then hand-reviewed; re-drawing it from
    # refreshed data would detach labels/gold_labels.csv from its items.
    if (LABEL_DIR / "gold_sample.csv").exists() and "--force" not in sys.argv:
        sys.exit(f"{LABEL_DIR / 'gold_sample.csv'} already exists and has been labelled; "
                 "use --force to draw a new sample.")
    LABEL_DIR.mkdir(exist_ok=True)
    (d, ds), (n, ns) = disclosure_items(), news_items()
    sample = pd.concat([d, n]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "item_id", [f"G{i:03d}" for i in range(1, len(sample) + 1)])
    sample["company"] = sample["symbol"].map(lambda s: STOCKS[s][1])
    sample.to_csv(LABEL_DIR / "gold_sample.csv", index=False, encoding="utf-8-sig")
    pd.concat([ds, ns]).to_csv(LABEL_DIR / "strata.csv", index=False)
    print(f"{len(sample)} items -> {LABEL_DIR / 'gold_sample.csv'}")
    print(sample.groupby(["source", "stratum"]).size().to_string())


if __name__ == "__main__":
    main()
