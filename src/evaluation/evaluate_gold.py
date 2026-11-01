"""Evaluate the sentiment models on the DFM gold set (UAE text).

Gold labels: AI-suggested (Claude, from the text only, blind to model output),
then reviewed and corrected by the author in src/app/review_gold_labels.py. Items labelled
"unclear" (garbled text) are excluded.

Models compared on exactly the same items:
  FinBERT (raw), FinBERT + domain rule, Loughran-McDonald, VADER.

Because the sample was stratified by FinBERT's predicted label (negatives are
rare), plain accuracy over the sample over-represents negatives. We therefore
also report a re-weighted accuracy: each stratum's accuracy weighted by that
stratum's share of the full population (labels/strata.csv). The 15 items
sampled specifically because the domain rule changed them are evaluated
separately and excluded from the re-weighted estimate.

Output: results/gold_eval.csv, results/gold_eval_rule_items.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from config import ROOT
from evaluation.evaluate_benchmarks import LABELS, predict_lm, predict_vader

LABEL_DIR = ROOT / "labels"
N_BOOT = 2000
SEED = 902


def weighted_accuracy(df: pd.DataFrame, pred: str, strata: pd.DataFrame) -> float:
    """Population-weighted accuracy across FinBERT-prediction strata, per source."""
    total, weight = 0.0, 0.0
    for (source, stratum), g in df.groupby(["source", "stratum"]):
        share = strata.loc[(strata["source"] == source) & (strata["label"] == stratum), "population_share"]
        if g.empty or share.empty:
            continue
        # Each source counts equally (disclosures and news are separate populations).
        w = float(share.iloc[0]) / df["source"].nunique()
        total += w * (g[pred] == g["gold"]).mean()
        weight += w
    return total / weight


def bootstrap_ci(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """95% bootstrap confidence interval for plain accuracy."""
    rng = np.random.default_rng(SEED)
    correct = (y_true.to_numpy() == y_pred.to_numpy()).astype(float)
    means = [rng.choice(correct, len(correct)).mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    sample = pd.read_csv(LABEL_DIR / "gold_sample.csv")
    gold = pd.read_csv(LABEL_DIR / "gold_labels.csv")[["item_id", "label", "ai_label", "changed"]]
    strata = pd.read_csv(LABEL_DIR / "strata.csv")
    df = sample.merge(gold.rename(columns={"label": "gold"}), on="item_id")

    # Annotation quality: how often the author changed the AI suggestion.
    changed = df["gold"] != df["ai_label"]
    print(f"Reviewed {len(df)} items; author changed {changed.sum()} AI suggestions ({changed.mean():.1%}); "
          f"AI-vs-author kappa = {cohen_kappa_score(df['ai_label'], df['gold']):.3f}")

    df = df[df["gold"] != "unclear"].copy()
    df["lm"] = predict_lm(df["text"])
    df["vader"] = predict_vader(df["text"])
    models = {"FinBERT": "finbert_raw", "FinBERT + domain rule": "finbert_rule",
              "Loughran-McDonald": "lm", "VADER": "vader"}

    main_set = df[df["stratum"] != "rule_flipped"]
    rows = []
    for scope, part in [("all", main_set), ("DFM disclosures", main_set[main_set["source"] == "DFM disclosure"]),
                        ("News headlines", main_set[main_set["source"] == "News headline"])]:
        for name, col in models.items():
            lo, hi = bootstrap_ci(part["gold"], part[col])
            rows.append({"scope": scope, "model": name, "n": len(part),
                         "accuracy": accuracy_score(part["gold"], part[col]), "acc_ci_low": lo, "acc_ci_high": hi,
                         "macro_f1": f1_score(part["gold"], part[col], labels=LABELS, average="macro"),
                         "weighted_accuracy": weighted_accuracy(part, col, strata) if scope == "all" else np.nan,
                         **{f"f1_{l}": v for l, v in zip(LABELS, f1_score(part["gold"], part[col], labels=LABELS,
                                                                           average=None))}})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "gold_eval.csv", index=False)

    # Items sampled because the domain rule flipped them: did the rule help?
    flipped = df[df["stratum"] == "rule_flipped"]
    rule = pd.DataFrame({"item_id": flipped["item_id"], "text": flipped["text"], "gold": flipped["gold"],
                         "finbert_raw": flipped["finbert_raw"], "finbert_rule": flipped["finbert_rule"]})
    rule.to_csv(ROOT / "results" / "gold_eval_rule_items.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\nAccuracy on the DFM gold set (unclear items excluded):")
    print(res[["scope", "model", "n", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1", "weighted_accuracy"]]
          .round(3).to_string(index=False))
    print(f"\nDomain-rule items ({len(rule)}): raw FinBERT correct {(rule.finbert_raw == rule.gold).sum()}, "
          f"with rule correct {(rule.finbert_rule == rule.gold).sum()}")
    print(rule[["gold", "finbert_raw", "finbert_rule", "text"]].assign(text=rule["text"].str.slice(0, 80))
          .to_string(index=False))


if __name__ == "__main__":
    main()
