"""Domain rule correcting a systematic FinBERT error: "a decrease is bad news".

FinBERT reads verbs like "cuts", "falls" or "dropped" as negative regardless of
*what* decreased. In our data this misclassifies good news, e.g.
  "DEWA's Smart Bank Guarantee System cuts turnaround time by 95%"  -> -0.82
  "NPL ratio has dropped from 6 to 5.6"                             -> -0.97
When a decrease applies to something undesirable (time, cost, losses, bad
loans, emissions, ...), a fall is good news, so we swap FinBERT's positive and
negative probabilities for that text.

The rule is only used because it improves (or does not hurt) accuracy on the
labelled benchmarks - see evaluate_benchmarks.py.
"""
import re

import pandas as pd

DECREASE = r"(?:cut|cuts|cutting|reduc\w*|lower\w*|fall|falls|fell|falling|drop\w*|declin\w*|decreas\w*|shrink\w*|" \
           r"slash\w*|halv\w*|sav(?:e|es|ed|ing))"
UNDESIRABLE = r"(?:time|turnaround|waiting|wait|delays?|costs?|expenses?|losses|loss|npls?|non-performing|" \
              r"impairments?|provisions?|emissions?|debt|leverage|complaints?|defaults?|outages?|downtime|" \
              r"cost-to-income|bad loans?|delinquenc\w*|carbon|fraud|refund time)"
# Words that signal the decrease applies to something else, e.g.
# "profit fell 20% due to higher provisions" (profit fell, provisions rose).
STOP = r"(?:due|because|after|amid|as|on|despite|while|but|higher|increas\w*|ris\w*|rose|up|growth)"
# Up to 3 words between the decrease and the undesirable quantity, in either
# order, separated by whitespace only (no clause boundary such as ; , .).
GAP = rf"(?:\s+(?!{STOP}\b)[\w%$.-]+){{0,3}}\s+"
PATTERN = re.compile(rf"\b{DECREASE}\b{GAP}{UNDESIRABLE}\b|\b{UNDESIRABLE}\b{GAP}(?:has\s+|have\s+|was\s+|were\s+)?{DECREASE}\b",
                     re.I)


def matches(text: str) -> bool:
    return bool(PATTERN.search(str(text)))


def apply(df: pd.DataFrame, text_col: str, prefix: str = "") -> pd.DataFrame:
    """Swap p_positive/p_negative where the rule fires; recompute polarity and label.

    The raw FinBERT polarity is kept in <prefix>polarity_raw.
    """
    pos, neg, neu = f"{prefix}p_positive", f"{prefix}p_negative", f"{prefix}p_neutral"
    out = df.copy()
    # Idempotent: undo flips from an earlier run before deciding again.
    flipped_col = f"{prefix}rule_flipped"
    if flipped_col in out:
        prev = out[flipped_col].fillna(False).astype(bool)
        out.loc[prev, [pos, neg]] = out.loc[prev, [neg, pos]].to_numpy()
    hit = out[text_col].map(matches) & (out[neg] > out[pos])
    out[f"{prefix}polarity_raw"] = out[pos] - out[neg]
    out.loc[hit, [pos, neg]] = out.loc[hit, [neg, pos]].to_numpy()
    out[f"{prefix}polarity"] = out[pos] - out[neg]
    out[f"{prefix}label"] = out[[pos, neg, neu]].idxmax(axis=1).str.replace(f"{prefix}p_", "", regex=False)
    out[f"{prefix}rule_flipped"] = hit
    return out
