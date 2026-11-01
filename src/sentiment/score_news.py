"""Score news headlines with FinBERT (Google News + English Mubasher).

Headlines are single sentences, very close in style to FinBERT's training
data (Financial PhraseBank), so they need no sentence splitting. Texts are
scored in batches sorted by length, which reduces padding and speeds up CPU
inference considerably.

Output: data/processed/news_sentiment.csv      (news.csv + probabilities, polarity, label)
        data/processed/mubasher_sentiment.csv  (English Mubasher headline and lead scored separately)
"""
import pandas as pd

from sentiment import domain_rules
from config import PROCESSED
from sentiment.score_documents import load_model, score


def score_texts(texts: pd.Series, tok, model, labels, name: str, apply_rule: bool = True) -> pd.DataFrame:
    """Score a Series of texts; returns probabilities aligned to its index.

    With apply_rule, the domain rule (domain_rules.py) corrects FinBERT's
    "a decrease is bad news" error; the raw score is kept in polarity_raw.
    """
    order = texts.str.len().sort_values().index
    values = texts.loc[order].tolist()
    chunks, step = [], 512
    for i in range(0, len(values), step):
        chunks.append(score(values[i:i + step], tok, model, labels))
        print(f"  {name}: scored {min(i + step, len(values))}/{len(values)}", flush=True)
    probs = pd.concat(chunks, ignore_index=True)
    probs.index = order
    probs = probs.sort_index()
    if apply_rule:
        probs = domain_rules.apply(probs.assign(_text=texts), "_text").drop(columns="_text")
    else:
        probs["polarity"] = probs["p_positive"] - probs["p_negative"]
        probs["label"] = probs[["p_positive", "p_negative", "p_neutral"]].idxmax(axis=1).str[2:]
    return probs


def main():
    tok, model, labels = load_model()

    news = pd.read_csv(PROCESSED / "news.csv")
    news = news.join(score_texts(news["headline"], tok, model, labels, "Google News"))
    news.to_csv(PROCESSED / "news_sentiment.csv", index=False, encoding="utf-8-sig")
    print(f"\nGoogle News: {len(news)} headlines\n{news['label'].value_counts().to_string()}\n")

    mub = pd.read_csv(PROCESSED / "mubasher.csv")
    mub = mub[mub["lang"] == "en"].copy()  # Arabic needs an Arabic model (future work)
    mub["lead"] = mub["lead"].fillna("")
    head = score_texts(mub["headline"], tok, model, labels, "Mubasher headlines").add_prefix("headline_")
    has_lead = mub["lead"].str.len() > 20
    lead = score_texts(mub.loc[has_lead, "lead"], tok, model, labels, "Mubasher leads").add_prefix("lead_")
    mub = mub.join(head).join(lead)
    mub.to_csv(PROCESSED / "mubasher_sentiment.csv", index=False, encoding="utf-8-sig")
    print(f"\nMubasher (EN): {len(mub)} items\n{mub['headline_label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
