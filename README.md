# Efsah Insight: Bilingual NLP for Dubai Financial Market Disclosures

*MAAI 902 – Natural Language Processing & LLMs, Assessment Task 2*

Efsah Insight reads the official company disclosures and news of Dubai Financial Market (DFM)
stocks, measures their sentiment with FinBERT, relates it to share-price behaviour, and presents a
transparent **Buy / Hold / Sell** view with a **bilingual (English / Arabic) explanation** grounded in
cited evidence and generated with **Falcon** (Technology Innovation Institute, Abu Dhabi).

> **Decision support only, not financial advice.** Back-tests show the signal's edge is small and
> unstable after trading costs. The value of the system is in making disclosures readable and
> traceable, not in predicting prices.

*"Efsah" (إفصاح) is Arabic for "disclosure"; DFM's disclosure data service uses the same word.*

## Contents

1. [What it does](#what-it-does)
2. [Repository layout](#repository-layout)
3. [Setup](#setup)
4. [Get the data and run the pipeline](#get-the-data-and-run-the-pipeline)
5. [Run the app](#run-the-app)
6. [Refresh the data](#refresh-the-data)
7. [Data sources](#data-sources)
8. [Key results](#key-results)
9. [Responsible AI](#responsible-ai)
10. [AI assistance](#ai-assistance)

---

## What it does

| Stage | What happens | Main tools |
|---|---|---|
| Collect | DFM disclosures (EN + AR PDFs, 2019–2026), daily prices, Google News headlines, Mubasher news, Emirates NBD results-call transcripts | `requests`, `yfinance`, `feedparser` |
| Extract | PDF text, OCR for scanned pages (English and Arabic), Arabic normalisation and lam-alef / split-word repair | PyMuPDF, Tesseract |
| Pre-process | Boilerplate removal, line re-joining, sentence segmentation, quality filter, entity disambiguation | Python `re` |
| Sentiment | Sentence-level FinBERT + a tested domain rule ("costs/bad loans fell" = good news) | Hugging Face Transformers |
| Evaluate | Benchmarks vs Loughran-McDonald and VADER; DFM gold set; Arabic cross-lingual comparison; event study | scikit-learn, SciPy, statsmodels |
| Signal | News tone + disclosure tone + price momentum → Buy / Hold / Sell, strength and agreement | pandas |
| Explain | Evidence retrieval (Chroma vector store) → Falcon-H1 writes EN explanation → AR translation with a quality gate | LangChain, sentence-transformers, Falcon-H1 |
| Present | Web app: exchange → stock → signal, chart, cited bilingual explanation, evaluation tabs | Streamlit, Plotly |

---

## Repository layout

```
├── src/                    # all code (run pipeline commands from this folder)
│   ├── config.py           # paths, stock list and settings
│   ├── collection/         # disclosures, PDFs, prices, news, transcripts
│   ├── text_processing/    # PDF text, OCR, cleaning, Arabic repair
│   ├── sentiment/          # FinBERT, domain rule, Arabic approaches
│   ├── evaluation/         # benchmarks, gold set, event study
│   ├── signals/            # Buy/Hold/Sell signal, evidence index, explanations
│   └── app/                # Streamlit web app (app.py) and gold-set review tool (review_gold_labels.py)
├── labels/                 # gold-set sample, AI-suggested labels, author-verified labels
├── results/                # evaluation outputs (CSV)
├── data/                   # NOT in git; created by the pipeline
│   ├── raw/                # downloaded data, as collected
│   └── processed/          # everything the pipeline generates
├── requirements.txt
└── README.md
```

---

## Setup

Tested on Windows 11 with Python 3.12. The commands below are for PowerShell. You need about 3 GB of free
disk space for the data, plus room for the models.

**1. Get the code**

```powershell
git clone https://github.com/RalanshaFernandes/MAAI902-IndividualProject-.git
cd MAAI902-IndividualProject-
```

**2. Install the packages** (PyTorch first, so you get the smaller CPU build)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**3. Install Tesseract OCR with Arabic support**

```powershell
winget install UB-Mannheim.TesseractOCR

$tess = "$HOME\models\tessdata"
New-Item -ItemType Directory -Force $tess
Copy-Item "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata", "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" $tess
Invoke-WebRequest https://github.com/tesseract-ocr/tessdata_fast/raw/main/ara.traineddata -OutFile "$tess\ara.traineddata"
```

The code expects Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If yours is elsewhere, change
`tesseract_cmd` in `src/text_processing/ocr_scanned_en.py` and `extract_text_ar.py`.

**4. Download Falcon** (optional)

Other models download automatically on first use. Falcon is only used if it is already on disk; without it,
explanations use the built-in template. From `src/`:

```powershell
cd src
python -m collection.download_model tiiuae/Falcon-H1-1.5B-Instruct
cd ..
```

This downloader can resume, which helps on slow connections. It also works for CAMeLBERT and OPUS-MT, using the
same command with `CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment`,
`CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment` or `Helsinki-NLP/opus-mt-ar-en`.

The 1.5B model runs on a laptop CPU at about 2 minutes per explanation. With a GPU, set
`$env:FALCON_MODEL = "tiiuae/Falcon-H1-7B-Instruct"` for better English and Arabic.

---

## Get the data and run the pipeline

`data/` is not in this repository, so **run these steps once before starting the app**. Run them from `src/`, in
this order, because each stage uses the output of the one before.

A full run takes several hours on a laptop CPU, mostly PDF download, OCR and model inference. Requests are
rate-limited (1–2 s apart). Downloads are cached, so if a step stops or DFM throttles you, wait and run it again.

```powershell
cd src
```

**Stage 1: Collect the data**

```powershell
python -m collection.collect_disclosures
python -m collection.prices
python -m collection.build_market_index
python -m collection.download_pdfs
python -m collection.download_pdfs --lang ar
python -m collection.collect_news
python -m collection.collect_mubasher
python -m collection.collect_transcripts
```

**Stage 2: Extract text from the PDFs**

```powershell
python -m text_processing.extract_text_en
python -m text_processing.ocr_scanned_en
python -m text_processing.extract_text_ar
```

**Stage 3: Score sentiment**

```powershell
python -m sentiment.score_documents
python -m sentiment.score_news
python -m sentiment.score_transcripts
python -m sentiment.compare_arabic_models
```

**Stage 4: Evaluate**

```powershell
python -m evaluation.evaluate_benchmarks
python -m evaluation.event_study
python -m evaluation.evaluate_gold
```

**Stage 5: Build the signal and explanations**

```powershell
python -m signals.signal_model
python -m signals.build_index
python -m signals.explain EMAAR EMIRATESNBD DIB DU DEWA SALIK
```

```powershell
cd ..
```

### What each step produces

| Stage | Module | Output |
|---|---|---|
| 1 | `collection.collect_disclosures` | `data/raw/disclosures.csv`, `attachments.csv`, `attachments_ar.csv` |
| 1 | `collection.prices` | `data/raw/prices/<SYMBOL>.csv`, daily prices for the 6 stocks |
| 1 | `collection.build_market_index` | `data/processed/market_index.csv`, equal-weighted DFM benchmark |
| 1 | `collection.download_pdfs` (and `--lang ar`) | `data/raw/pdfs/`, English and Arabic disclosure PDFs |
| 1 | `collection.collect_news` | `data/raw/news/`, `data/processed/news.csv` |
| 1 | `collection.collect_mubasher` | `data/raw/mubasher/`, `data/processed/mubasher.csv` |
| 1 | `collection.collect_transcripts` | `data/raw/transcripts/`, `data/processed/transcript_turns.csv` |
| 2 | `text_processing.extract_text_en` | `data/processed/text_en/`, `documents.csv` |
| 2 | `text_processing.ocr_scanned_en` | OCR text for scanned English PDFs |
| 2 | `text_processing.extract_text_ar` | `data/processed/text_ar/`, `documents_ar.csv` |
| 3 | `sentiment.score_documents` | `sentences.csv`, `doc_sentiment.csv` |
| 3 | `sentiment.score_news` | `news_sentiment.csv`, `mubasher_sentiment.csv` |
| 3 | `sentiment.score_transcripts` | `transcript_sentences.csv`, `transcript_tone.csv` |
| 3 | `sentiment.compare_arabic_models` | `arabic_docs.csv`, `results/arabic_crosslingual.csv` |
| 4 | `evaluation.evaluate_benchmarks` | `results/model_eval.csv`, `confusion_matrices.csv`, `model_agreement.csv` |
| 4 | `evaluation.event_study` | `events_*.csv`, `results/event_study_summary.csv` |
| 4 | `evaluation.evaluate_gold` | `results/gold_eval.csv`, `gold_eval_rule_items.csv` |
| 5 | `signals.signal_model` | `signals.csv`, `results/signal_classes.csv`, `backtest_summary.csv`, `backtest_robustness.csv` |
| 5 | `signals.build_index` | `evidence.csv`, Chroma index in `~/.cache/dfm_nlp/index` |
| 5 | `signals.explain <SYMBOL> ...` | `data/processed/explanations/`, one explanation per stock |

### The gold set

The hand-reviewed gold set is already in `labels/`, so you do not need to rebuild it. **Do not run
`evaluation.make_gold_sample --force`**: it replaces the sample the reviewed labels belong to. The tools used to
create it are `evaluation.make_gold_sample`, `evaluation.write_ai_labels` and `src/app/review_gold_labels.py`.

### Optional settings

| Environment variable | Purpose |
|---|---|
| `FALCON_MODEL` | Falcon model to use (default `tiiuae/Falcon-H1-1.5B-Instruct`) |
| `DFM_PDF_DIR` | Keep the PDFs (about 1.5 GB) outside the project, for example outside a cloud-synced folder |
| `DFM_INDEX_DIR` | Where to store the evidence index (default `~/.cache/dfm_nlp/index`) |
| `GOLD_LABELS_FILE` | Label file used by the review tool (default `labels/gold_labels.csv`) |

---

## Run the app

Run these from the project root, after completing the pipeline.

```powershell
streamlit run src/app/app.py
```

The app opens at http://localhost:8501. To open the gold-set review tool:

```powershell
streamlit run src/app/review_gold_labels.py --server.port 8502
```

---

## Refresh the data

Refreshing is incremental. Downloads, text extraction, OCR and FinBERT document scoring skip anything already
processed.

1. Delete the current month's files in `data/raw/news/<SYMBOL>/`, and the Mubasher pages in `data/raw/mubasher/`,
   because they hold the state of the last run.
2. Re-run the stages above, leaving out `collection.download_pdfs --lang ar`, `text_processing.extract_text_ar`,
   `sentiment.compare_arabic_models`, `evaluation.evaluate_benchmarks` and `evaluation.evaluate_gold`.
   The benchmarks and gold set do not depend on new data.

To force a full re-run, add `--all` to `text_processing.extract_text_en` or `sentiment.score_documents`.

---

## Data sources

| Source | Volume | Use |
|---|---|---|
| DFM disclosures (DFM's `efsah` data API) | 3,075 disclosures; 1,669 English + 1,536 Arabic PDFs since 2019 | main text source |
| Yahoo Finance | daily prices, 6 stocks (+ up to 64 a day for the market index) | momentum, event study |
| Google News search feeds | 7,126 headlines | independent media view |
| Mubasher | 1,636 items | second news source |
| Emirates NBD investor relations | 34 results-call transcripts | management vs analyst tone |
| Financial PhraseBank, Twitter Financial News | 970 + 2,388 labelled texts | model benchmarks |

Stocks: Emaar Properties, Emirates NBD, Dubai Islamic Bank, du, DEWA, Salik. Data last refreshed on
28 October 2026. The Arabic PDFs, cross-lingual evaluation and gold set were built from the data available on
9 October 2026 and were not extended, so your numbers will differ if you collect the data again.

Raw data is not redistributed in this repository. For news, only metadata, headlines, leads and links are stored
(no full articles). The DFM data service used is undocumented: requests are rate-limited and cached, and any
production use would require a licensed data feed.

---

## Key results

Details are in `results/` and the report (data to 28 October 2026).

- **Corporate positivity bias.** Company-controlled text is far more positive than independent text.
  Polarity: press releases +0.46, management's prepared remarks +0.49, media headlines +0.31,
  analysts' questions +0.08.
- **Model choice.** On unseen labelled financial news (Twitter Financial News), FinBERT scores macro-F1 0.67,
  against 0.45 for both the Loughran-McDonald dictionary and VADER.
- **Error analysis.** FinBERT reads "costs fell" or "NPL ratio dropped" as bad news. A narrow, tested rule fixes
  this without hurting benchmark accuracy.
- **Entity disambiguation.** "Dewa United" (a football club) and "Emaar The Economic City" (a separate
  Saudi company) polluted the news data and are now filtered out.
- **Prices.** Disclosure and news sentiment do not significantly predict short-term abnormal returns.
  Emirates NBD management tone on results calls does (exploratory, n = 27).
- **Signal.** BUY beats SELL in 2023–2026, but combining sentiment with momentum is only marginally ahead of
  momentum alone before costs, and that ranking is unstable. After costs the combined signal is slightly
  negative (-1.0% a year). Hence the decision-support framing.

---

## Responsible AI

- **No look-ahead.** Only text published before each trading session is used.
- **Strength, not "confidence".** Signal strength is shown as Low / Medium / High with indicator agreement.
  A percentage would imply a probability the back-tests do not support.
- **Grounded explanations.** Falcon may only use retrieved, cited evidence. Citations and numbers are checked
  automatically, Arabic output passes a quality gate, and a deterministic template is the fallback.
- **Missing data is treated as missing,** not as neutral.
- **Limitations.** Sentiment is primarily English. No established Arabic-specific financial sentiment model was
  found, so Arabic approaches are compared cross-lingually. The gold set uses AI-suggested labels verified by the
  author.

---

## AI assistance

This project was developed with the help of an AI coding assistant (Claude). Gold-set labels were suggested
by the assistant from the text alone and reviewed and corrected by the author.
