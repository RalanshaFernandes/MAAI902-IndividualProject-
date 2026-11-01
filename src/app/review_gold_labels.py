"""Review tool for the DFM gold set (AI-suggested labels, human-verified).

Run:  streamlit run src/app/review_gold_labels.py --server.port 8502

Each item shows the AI annotator's suggested label (labels/ai_labels.csv,
written by src/evaluation/write_ai_labels.py from the text only). The author accepts it
("Agree") or picks a different label. Every click is saved to
labels/gold_labels.csv with the AI suggestion and whether it was changed, so
the report can state how many suggestions the author corrected.
The model's (FinBERT's) predictions are never shown.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
import os

SAMPLE = ROOT / "labels" / "gold_sample.csv"
AI = ROOT / "labels" / "ai_labels.csv"
# GOLD_LABELS_FILE lets automated tests write elsewhere and never touch real labels.
LABELS = Path(os.environ.get("GOLD_LABELS_FILE", ROOT / "labels" / "gold_labels.csv"))
CHOICES = [("positive", "👍 Positive"), ("neutral", "➖ Neutral"), ("negative", "👎 Negative"), ("unclear", "❓ Unclear")]
NAMES = dict(CHOICES)

st.set_page_config(page_title="DFM Label Review", page_icon="🏷️", layout="centered")

sample = pd.read_csv(SAMPLE)[["item_id", "source", "company", "text", "context"]]   # model columns hidden
sample = sample.merge(pd.read_csv(AI), on="item_id")
labels = pd.read_csv(LABELS) if LABELS.exists() else pd.DataFrame(
    columns=["item_id", "label", "ai_label", "changed", "labelled_at"])
# Labels made in the earlier blind version have no AI-suggestion columns: keep them, fill the gaps.
if "ai_label" not in labels:
    labels = labels.merge(pd.read_csv(AI), on="item_id", how="left")
if "changed" not in labels:
    labels["changed"] = labels["label"] != labels["ai_label"]
done = dict(zip(labels["item_id"], labels["label"]))


def save(item: pd.Series, label: str) -> None:
    global labels
    labels = labels[labels["item_id"] != item["item_id"]]
    labels = pd.concat([labels, pd.DataFrame([{
        "item_id": item["item_id"], "label": label, "ai_label": item["ai_label"],
        "changed": label != item["ai_label"], "labelled_at": datetime.now().isoformat(timespec="seconds")}])])
    tmp = LABELS.with_suffix(".tmp")
    labels.to_csv(tmp, index=False)
    tmp.replace(LABELS)   # atomic write: a crash never leaves a half-written file


if "pos" not in st.session_state:   # start at the first unreviewed item
    todo = [i for i, r in sample.iterrows() if r["item_id"] not in done]
    st.session_state.pos = todo[0] if todo else len(sample) - 1

with st.sidebar:
    st.header("How to review")
    st.markdown("""
Each sentence shows a **suggested label**. Imagine you **own shares in the company**:
if this were news today, is it good, bad or neither for the company's value?

- **✓ Agree** if the suggestion is right (most of the time)
- otherwise click the label you think is correct

**👍 Positive**: profit up, dividend, new contract, costs / bad loans *down*, upgrade
**👎 Negative**: profit down, losses, provisions up, delays, downgrade, fines, lawsuits
**➖ Neutral**: notices, names, dates, accounting policy text, table rows
**❓ Unclear**: garbled text, or you genuinely can't tell

Take a few seconds each; go with your judgement.
""")
    st.caption(f"Saving to {LABELS}")

n_done = sum(i in done for i in sample["item_id"])
n_changed = int(labels["changed"].astype(bool).sum()) if len(labels) else 0
st.progress(n_done / len(sample), text=f"{n_done} of {len(sample)} reviewed · {n_changed} changed")
if n_done == len(sample):
    st.success("All items reviewed. Thank you! You can still go back and change any label.")

pos = st.session_state.pos
item = sample.iloc[pos]
st.caption(f"Item {item['item_id']} · {item['source']} · {item['company']}")
st.markdown(f"#### {item['text']}")
st.caption(f"Context: {item['context']}")
st.markdown(f"Suggested label: **{NAMES[item['ai_label']]}**")
if item["item_id"] in done:
    st.info(f"Your label: **{NAMES[done[item['item_id']]]}**")

if st.button(f"✓ Agree: {NAMES[item['ai_label']]}", type="primary", width="stretch", key=f"agree_{pos}"):
    save(item, item["ai_label"])
    st.session_state.pos = min(pos + 1, len(sample) - 1)
    st.rerun()

st.caption("…or choose a different label:")
cols = st.columns(4)
for col, (value, text) in zip(cols, CHOICES):
    if col.button(text, width="stretch", key=f"{value}_{pos}", disabled=value == item["ai_label"]):
        save(item, value)
        st.session_state.pos = min(pos + 1, len(sample) - 1)
        st.rerun()

nav = st.columns(3)
if nav[0].button("← Back", disabled=pos == 0):
    st.session_state.pos = pos - 1
    st.rerun()
if nav[1].button("Next unreviewed →"):
    remaining = [i for i, r in sample.iterrows() if r["item_id"] not in done]
    if remaining:
        st.session_state.pos = remaining[0]
        st.rerun()
if nav[2].button("Skip →", disabled=pos == len(sample) - 1):
    st.session_state.pos = pos + 1
    st.rerun()
