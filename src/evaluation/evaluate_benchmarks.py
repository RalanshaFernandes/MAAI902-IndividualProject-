"""How well do the sentiment models read financial text?

Compares three approaches on labelled data:
  * FinBERT (ProsusAI/finbert)  - transformer fine-tuned for financial sentiment
  * Loughran-McDonald (LM)      - finance-specific word lists (dictionary baseline)
  * VADER                       - general-purpose sentiment rules (non-financial baseline)

Datasets:
  * Financial PhraseBank (Malo et al., 2014), test split of the atrost/financial_phrasebank
    copy (970 sentences). CAUTION: FinBERT was fine-tuned on Financial PhraseBank, so
    its score here is optimistic (possible train/test overlap).
  * Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment),
    validation split (2,388 texts). FinBERT has not seen this data -> fairer test.

Also measures how often the models agree on our own Google News headlines,
which have no gold labels.

Output: results/model_eval.csv, results/confusion_matrices.csv, results/model_agreement.csv
"""
import pandas as pd
import pysentiment2 as ps
from datasets import load_dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import PROCESSED, ROOT
from sentiment.score_news import score_texts
from sentiment.score_documents import load_model

LABELS = ["negative", "neutral", "positive"]


def load_benchmarks() -> dict[str, pd.DataFrame]:
    fpb = load_dataset("atrost/financial_phrasebank")["test"].to_pandas()
    names = load_dataset("atrost/financial_phrasebank")["test"].features["label"].names
    fpb = pd.DataFrame({"text": fpb["sentence"], "label": fpb["label"].map(dict(enumerate(names)))})

    tw = load_dataset("zeroshot/twitter-financial-news-sentiment")["validation"].to_pandas()
    tw = pd.DataFrame({"text": tw["text"], "label": tw["label"].map({0: "negative", 1: "positive", 2: "neutral"})})
    return {"Financial PhraseBank (test)": fpb, "Twitter Financial News (val)": tw}


def predict_lm(texts: pd.Series) -> pd.Series:
    """LM polarity = (pos - neg) / (pos + neg); no finance words -> neutral."""
    lm = ps.LM()
    pol = texts.map(lambda t: lm.get_score(lm.tokenize(t))["Polarity"])
    return pol.map(lambda p: "positive" if p > 0 else "negative" if p < 0 else "neutral")


def predict_vader(texts: pd.Series) -> pd.Series:
    """Standard VADER thresholds on the compound score (+/-0.05)."""
    v = SentimentIntensityAnalyzer()
    comp = texts.map(lambda t: v.polarity_scores(t)["compound"])
    return comp.map(lambda c: "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")


def predict_finbert(texts: pd.Series, tok, model, labels, name: str, apply_rule: bool) -> pd.Series:
    return score_texts(texts.reset_index(drop=True), tok, model, labels, name,
                       apply_rule=apply_rule)["label"].set_axis(texts.index)


def main():
    tok, model, labels = load_model()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows, cms = [], []

    for ds_name, df in load_benchmarks().items():
        preds = {"FinBERT": predict_finbert(df["text"], tok, model, labels, ds_name, apply_rule=False),
                 "FinBERT + domain rule": predict_finbert(df["text"], tok, model, labels, ds_name, apply_rule=True),
                 "Loughran-McDonald": predict_lm(df["text"]),
                 "VADER": predict_vader(df["text"])}
        for model_name, pred in preds.items():
            f1 = f1_score(df["label"], pred, labels=LABELS, average=None)
            rows.append({"dataset": ds_name, "model": model_name, "n": len(df),
                         "accuracy": accuracy_score(df["label"], pred),
                         "macro_f1": f1_score(df["label"], pred, labels=LABELS, average="macro"),
                         **{f"f1_{l}": v for l, v in zip(LABELS, f1)}})
            cm = pd.DataFrame(confusion_matrix(df["label"], pred, labels=LABELS),
                              index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
            cms.append(cm.assign(dataset=ds_name, model=model_name).reset_index(names="true"))

    res = pd.DataFrame(rows)
    res.to_csv(out / "model_eval.csv", index=False)
    pd.concat(cms).to_csv(out / "confusion_matrices.csv", index=False)

    # Agreement on our own (unlabelled) news headlines.
    news = pd.read_csv(PROCESSED / "news_sentiment.csv")
    lm_news, vader_news = predict_lm(news["headline"]), predict_vader(news["headline"])
    agree = pd.DataFrame([
        {"pair": "FinBERT vs LM", "agreement": (news["label"] == lm_news).mean(),
         "cohen_kappa": cohen_kappa_score(news["label"], lm_news)},
        {"pair": "FinBERT vs VADER", "agreement": (news["label"] == vader_news).mean(),
         "cohen_kappa": cohen_kappa_score(news["label"], vader_news)},
        {"pair": "LM vs VADER", "agreement": (lm_news == vader_news).mean(),
         "cohen_kappa": cohen_kappa_score(lm_news, vader_news)},
    ])
    agree.to_csv(out / "model_agreement.csv", index=False)

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print("\nAgreement on 6,967 Google News headlines (no gold labels):")
    print(agree.round(3).to_string(index=False))
    print("\nLabel distribution on news headlines:")
    print(pd.DataFrame({"FinBERT": news["label"].value_counts(normalize=True),
                        "LM": lm_news.value_counts(normalize=True),
                        "VADER": vader_news.value_counts(normalize=True)}).round(3).to_string())


if __name__ == "__main__":
    main()
