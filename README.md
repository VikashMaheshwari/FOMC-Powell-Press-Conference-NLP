# 🏦 FOMC Powell Press Conference — NLP Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

Comprehensive NLP analysis of 40 Federal Open Market Committee (FOMC) press conference transcripts delivered by Chair Jerome Powell from 2020 to 2025. Extracts Q&A patterns, tracks hawkish/dovish sentiment over the full rate cycle, and discovers latent communication themes via LDA topic modeling — deployed as an interactive Streamlit dashboard.

---

## Dataset

| Field | Value |
|-------|-------|
| Source | 40 FOMC press conference transcripts (2020–2025) |
| Format | `.txt` files in `archive.zip` |
| Coverage | 2020–2025 (full COVID era + rate hike cycle) |
| Q&A pairs | 1,098 (Chair Powell answers) |
| Mean answer | ~190 words |

---

## Analysis Pipeline

**1. Corpus EDA** — Word count stats, transcript timeline, Q&A extraction  
**2. Q&A Extraction** — Regex-based isolation of 1,098 Powell answers from XML tags  
**3. Word Frequency** — Top-N vocabulary with hawkish/dovish term highlighting  
**4. Sentiment Analysis (VADER)** — Per-conference compound score; rolling trend  
**5. Topic Modeling (LDA)** — 5 latent topics with heatmap over time  
**6. N-gram Language Model** — Trigram generator as a statistical baseline  

---

## Key Findings

| Finding | Detail |
|---------|--------|
| #1 substantive term | *inflation* (1,628 mentions) |
| Most hawkish period | 2022–2023 (fastest tightening since 1980s) |
| Most dovish period | 2020–2021 (COVID-era accommodation) |
| Dominant 2022 topic | Inflation & Prices (~40% topic weight) |
| Persistent theme | Labor Market & Employment (all years) |
| 2023 spike | Financial Stability (SVB / banking stress) |

---

## Sentiment Trajectory

```
2020–2021  →  DOVISH   — emergency cuts, QE, "support the economy at all costs"
2022–2023  →  HAWKISH  — fastest rate hike cycle since 1980s
2024–2025  →  NEUTRAL  — "data dependent", pivot signaling, soft-landing debate
```

---

## Tech Stack

`Python` · `Pandas` · `scikit-learn (LDA)` · `VADER Sentiment` · `Plotly` · `Streamlit` · `re (Regex)`

---

## Run Locally

```bash
git clone https://github.com/VikashMaheshwari/FOMC-Powell-Press-Conference-NLP.git
cd FOMC-Powell-Press-Conference-NLP
pip install -r requirements.txt

# Notebook
jupyter notebook FOMC_Powell_NLP_Analysis.ipynb

# Interactive dashboard
streamlit run app.py
```

> Place `archive.zip` (40 transcripts) in the project root. The Streamlit app also accepts the file via browser upload.

---

## Project Structure

```
FOMC-Powell-Press-Conference-NLP/
├── FOMC_Powell_NLP_Analysis.ipynb   # Full analysis notebook
├── app.py                            # Streamlit interactive dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Author

**Vikash Maheshwari** · M.Eng CS&E  
[Portfolio](https://vikash-maheshwari.vercel.app/) · [LinkedIn](https://linkedin.com/in/vikashmaheshwari) · [GitHub](https://github.com/VikashMaheshwari)
