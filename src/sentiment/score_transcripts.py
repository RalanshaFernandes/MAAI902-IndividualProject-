"""Score Emirates NBD call transcripts with FinBERT, sentence by sentence.

Each speaker turn is split into sentences with the same cleaning used for
disclosures; every sentence keeps its call, speaker, role and section, so tone
can be compared between management's prepared remarks, management's answers
and analysts' questions.

Output: data/processed/transcript_sentences.csv  one row per scored sentence
        data/processed/transcript_tone.csv       mean polarity per call x role x section
"""
import pandas as pd

from config import PROCESSED
from sentiment.score_news import score_texts
from text_processing.clean_english import sentences
from sentiment.score_documents import load_model


def main():
    turns = pd.read_csv(PROCESSED / "transcript_turns.csv")
    rows = []
    for t in turns.itertuples():
        for i, s in enumerate(sentences(t.text)):
            rows.append({"period": t.period, "call_date": t.call_date, "turn": t.turn, "speaker": t.speaker,
                         "role": t.role, "section": t.section, "sent_idx": i, "sentence": s})
    sents = pd.DataFrame(rows)
    sents = sents[sents["role"] != "operator"].reset_index(drop=True)
    print(f"{len(sents)} sentences from {sents['period'].nunique()} calls")

    tok, model, labels = load_model()
    sents = sents.join(score_texts(sents["sentence"], tok, model, labels, "Transcripts"))
    sents.to_csv(PROCESSED / "transcript_sentences.csv", index=False, encoding="utf-8-sig")

    tone = (sents.groupby(["period", "call_date", "section", "role"])
            .agg(sentences=("polarity", "size"), polarity=("polarity", "mean"),
                 share_negative=("label", lambda x: (x == "negative").mean()))
            .reset_index())
    tone.to_csv(PROCESSED / "transcript_tone.csv", index=False)
    print(sents.groupby(["section", "role"]).agg(sentences=("polarity", "size"), polarity=("polarity", "mean"),
                                                 share_negative=("label", lambda x: (x == "negative").mean()))
          .round(3).to_string())


if __name__ == "__main__":
    main()
