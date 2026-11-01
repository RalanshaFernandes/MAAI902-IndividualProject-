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
NLP/
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
├── report/                 # NOT in git; technical report (APA 7, .docx)
├── data/                   # NOT in git; rebuilt by the scripts
│   ├── raw/                # downloaded data, as collected
│   └── processed/          # everything the pipeline generates
├── requirements.txt
└── README.md
```

---

## Setup

Tested on Windows 11 with Python 3.12.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of PyTorch
```

External pieces:

- **Tesseract OCR**: `winget install UB-Mannheim.TesseractOCR`. For Arabic OCR, download
  [`ara.traineddata`](https://github.com/tesseract-ocr/tessdata_fast) into `~/models/tessdata/`
  together with copies of `eng.traineddata` and `osd.traineddata`.
- **Models** are fetched from Hugging Face on first use. `python -m collection.download_model <repo>` (from `src/`) is a resumable
  downloader into `~/models/` for slow or unstable connections (used for Falcon-H1, CAMeLBERT and opus-mt).
- **Falcon model size**: the default is `tiiuae/Falcon-H1-1.5B-Instruct` (laptop CPU, about 2 min per
  explanation). On a GPU set `FALCON_MODEL=tiiuae/Falcon-H1-7B-Instruct` for better English and Arabic.

---

## Running the pipeline

Run every step from `src/` as a module, for example `python -m collection.prices`. Each one writes its
outputs to `data/` or `results/` and can be re-run.

| # | Module | Output |
|---|---|---|
| 1 | `collection.collect_disclosures` | disclosure metadata, EN/AR headlines, attachment lists |
| 2 | `collection.prices` | daily prices for the 6 stocks |
| 3 | `collection.build_market_index` | equal-weighted DFM market index (all DFM equities with Yahoo prices) |
| 4 | `collection.download_pdfs` (and `--lang ar`) | English and Arabic disclosure PDFs |
| 5 | `text_processing.extract_text_en`, `text_processing.ocr_scanned_en` | English text, OCR for scanned PDFs |
| 6 | `sentiment.score_documents` | FinBERT sentence scores and document scores |
| 7 | `collection.collect_news`, `collection.collect_mubasher`, `collection.collect_transcripts` | news headlines, Mubasher items, call transcripts |
| 8 | `sentiment.score_news`, `sentiment.score_transcripts` | FinBERT on headlines and transcripts |
| 9 | `evaluation.evaluate_benchmarks` | FinBERT vs baselines on public benchmarks |
| 10 | `evaluation.event_study` | price reaction around events |
| 11 | `signals.signal_model` | Buy / Hold / Sell signals and back-test |
| 12 | `signals.build_index` | evidence vector index for explanations |
| 13 | `signals.explain <SYMBOL>` | bilingual explanation for a stock |
| 14 | `text_processing.extract_text_ar`, `sentiment.compare_arabic_models` | Arabic text and cross-lingual comparison |
| 15 | `evaluation.make_gold_sample`, `evaluation.write_ai_labels`, `evaluation.evaluate_gold` | gold-set sample, AI-suggested labels, evaluation on the reviewed labels |

Collecting and processing everything from scratch takes several hours on a laptop CPU, mostly PDF
download, OCR and model inference. Network calls are rate-limited (1–2 s between requests).

**Refreshing the data** is incremental: re-run steps 1–8 and 10–13 in order. Downloads, text extraction, OCR
and FinBERT document scoring skip anything already processed (`--all` forces a full re-run of
`text_processing.extract_text_en` and `sentiment.score_documents`). Delete the current month's cache files in
`data/raw/news/` and `data/raw/mubasher/` first, because they hold the state of the last run. The gold sample is
fixed: `evaluation.make_gold_sample` refuses to overwrite it without `--force`.

## Running the app

```bash
streamlit run src/app/app.py                       # http://localhost:8501
streamlit run src/app/review_gold_labels.py --server.port 8502  # gold-set review tool
```

---

## Data

| Source | Volume | Use |
|---|---|---|
| DFM disclosures (DFM's `efsah` data API) | 3,075 disclosures; 1,669 English + 1,536 Arabic PDFs since 2019 | main text source |
| Yahoo Finance | daily prices, 6 stocks (+ about 65 for the market index) | momentum, event study |
| Google News search feeds | 7,126 headlines | independent media view |
| Mubasher | 1,636 items | second news source |
| Emirates NBD investor relations | 34 results-call transcripts | management vs analyst tone |
| Financial PhraseBank, Twitter Financial News | 970 + 2,388 labelled texts | model benchmarks |

Stocks: Emaar Properties, Emirates NBD, Dubai Islamic Bank, du, DEWA, Salik. Data last refreshed on
28 October 2026, and the report uses that data. The Arabic PDFs, cross-lingual evaluation and gold set were built
from the data available on 9 October 2026 and were not extended.

Raw data is not redistributed in this repository. Only metadata, headlines and links are stored for news
(no full articles). The DFM data service used is undocumented: requests are rate-limited and cached, and
any production use would require a licensed data feed.

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

## Responsible AI

- **No look-ahead.** Only text published before each trading session is used.
- **Strength, not "confidence".** Signal strength is shown as Low / Medium / High with indicator agreement.
  A percentage would imply a probability the back-tests do not support.
- **Grounded explanations.** Falcon may only use retrieved, cited evidence. Citations and numbers are checked
  automatically, Arabic output passes a quality gate, and a deterministic template is the fallback.
- **Missing data is treated as missing,** not as neutral.
- **Limitations.** Sentiment is primarily English. No Arabic financial sentiment model exists, so Arabic
  approaches are compared cross-lingually. The gold set uses AI-suggested labels verified by the author.

## AI assistance

This project was developed with the help of an AI coding assistant (Claude). Gold-set labels were suggested
by the assistant from the text alone and reviewed and corrected by the author.
