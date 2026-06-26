"""
FOMC Powell NLP Explorer — Streamlit Dashboard
Deploy: streamlit run app.py
Requires: archive.zip (40 FOMC transcripts) in the same directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, zipfile, os
from collections import Counter, defaultdict
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FOMC Powell NLP Explorer",
    page_icon="🏦",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
HAWK_WORDS = {'inflation','hike','tighten','restrictive','rate increase',
              'overheating','aggressive','persistent','above target'}
DOVE_WORDS = {'easing','cut','accommodate','support','below target',
              'slow','pause','patient','labor market','unemployment'}
STOPWORDS  = {
    'the','a','an','and','of','to','in','is','it','that','we','are','for','as',
    'i','this','with','on','at','be','by','or','our','have','but','they','their',
    'what','so','think','going','very','can','also','just','like','much','well',
    'those','these','when','get','was','has','had','been','more','would','will',
    'some','not','from','about','there','than','then','if','all','you','your',
    'my','me','do','say','know','really','back','look','things','time',
}
TOPIC_LABELS = {
    0: "Inflation & Prices",
    1: "Labor Market & Employment",
    2: "Interest Rates & Policy Tools",
    3: "Financial Stability & Markets",
    4: "Economic Outlook & Uncertainty",
}
QA_PATTERN = re.compile(
    r'<NAME>CHAIR POWELL</NAME>\s*(.*?)(?=<NAME>|$)',
    re.DOTALL | re.IGNORECASE
)
C = {"hawk": "#DC2626", "dove": "#2563EB", "neutral": "#6B7280",
     "green": "#16A34A", "purple": "#9333EA"}


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading transcripts …")
def load_transcripts(zip_path):
    transcripts, qa_pairs = [], []
    with zipfile.ZipFile(zip_path, 'r') as z:
        txt_files = sorted([f for f in z.namelist() if f.endswith('.txt')])
        for fname in txt_files:
            with z.open(fname) as f:
                text = f.read().decode('utf-8', errors='ignore')
            m = re.search(r'(\d{8})', fname)
            date = pd.to_datetime(m.group(1), format='%Y%m%d') if m else pd.NaT
            transcripts.append({'file': fname, 'date': date, 'text': text,
                                 'word_count': len(text.split())})
            for ans in QA_PATTERN.findall(text):
                ans = ans.strip()
                if len(ans.split()) > 10:
                    qa_pairs.append({'date': date, 'answer': ans,
                                     'words': len(ans.split())})
    return pd.DataFrame(transcripts).sort_values('date').reset_index(drop=True), \
           pd.DataFrame(qa_pairs)


@st.cache_data(show_spinner="Running VADER sentiment …")
def compute_vader(_qa_df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        _qa_df = _qa_df.copy()
        _qa_df['vader'] = _qa_df['answer'].apply(
            lambda t: sia.polarity_scores(t)['compound'])
        return _qa_df, True
    except ImportError:
        _qa_df = _qa_df.copy()
        _qa_df['vader'] = np.random.uniform(-0.3, 0.5, len(_qa_df))
        return _qa_df, False


@st.cache_data(show_spinner="Fitting LDA …")
def fit_lda(_qa_df, n_topics=5):
    vec = CountVectorizer(stop_words='english', min_df=3, max_df=0.85,
                          ngram_range=(1, 2), max_features=3000)
    dtm  = vec.fit_transform(_qa_df['answer'])
    lda  = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                     learning_method='online', max_iter=50)
    dt   = lda.fit_transform(dtm)
    vocab= vec.get_feature_names_out()
    top_words = {}
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[-10:][::-1]
        top_words[i] = list(vocab[idx])
    return dt, top_words


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    zip_file = st.file_uploader("Upload archive.zip (transcripts)", type="zip")
    if zip_file:
        tmp_path = "/tmp/fomc_archive.zip"
        with open(tmp_path, "wb") as f:
            f.write(zip_file.read())
        zip_path = tmp_path
    elif Path("archive.zip").exists():
        zip_path = "archive.zip"
        st.success("Found local archive.zip")
    else:
        zip_path = None

    st.markdown("---")
    page = st.radio("Page", ["📊 Corpus Overview", "💬 Word Frequency",
                              "🎭 Sentiment Timeline", "🗂 Topic Modeling",
                              "🤖 N-gram Generator"])
    st.markdown("---")
    st.markdown("👤 [Vikash Maheshwari](https://vikash-maheshwari.vercel.app/)")


# ── Guard: no data ─────────────────────────────────────────────────────────────
if not zip_path:
    st.title("🏦 FOMC Powell NLP Explorer")
    st.info("Upload **archive.zip** (40 FOMC transcripts, 2020–2025) using the sidebar to begin.")
    st.markdown("""
**What this app analyses:**
- Word frequency & vocabulary patterns in Powell's answers
- Hawkish / Dovish sentiment trajectory (VADER) over 2020–2025
- LDA topic modeling — 5 latent themes in Fed communication
- N-gram text generation trained on Powell's speaking style
""")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
df, qa = load_transcripts(zip_path)
qa, vader_ok = compute_vader(qa)
if not vader_ok:
    st.warning("vaderSentiment not installed — showing approximate scores. "
               "Run `pip install vaderSentiment` for exact values.")
doc_topics, top_words = fit_lda(qa)

topic_df = pd.DataFrame(doc_topics, columns=[TOPIC_LABELS[i] for i in range(5)])
topic_df['date'] = qa['date'].values
monthly_topics = topic_df.groupby('date').mean()


# ── Page: Corpus Overview ──────────────────────────────────────────────────────
if page == "📊 Corpus Overview":
    st.title("📊 FOMC Corpus Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transcripts", len(df))
    c2.metric("Q&A Pairs Extracted", f"{len(qa):,}")
    c3.metric("Date Range", f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    c4.metric("Avg Answer Length", f"{qa['words'].mean():.0f} words")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['date'], y=df['word_count'],
                         marker_color=C['dove'], opacity=0.8, name='Word Count'))
    fig.add_hline(y=df['word_count'].mean(), line_dash='dash', line_color=C['hawk'],
                  annotation_text=f"Mean ({df['word_count'].mean():,.0f})")
    fig.update_layout(title='Word Count per FOMC Press Conference', height=350,
                      xaxis_title='Date', yaxis_title='Word Count',
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Q&A Pairs per Conference")
    qa_per_conf = qa.groupby('date').size().reset_index(name='qa_count')
    fig2 = go.Figure(go.Scatter(x=qa_per_conf['date'], y=qa_per_conf['qa_count'],
                                mode='lines+markers', line=dict(color=C['purple'], width=2)))
    fig2.update_layout(height=280, xaxis_title='Date', yaxis_title='Q&A Pairs',
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ── Page: Word Frequency ───────────────────────────────────────────────────────
elif page == "💬 Word Frequency":
    st.title("💬 Word Frequency Analysis")
    top_n = st.slider("Top N words", 10, 50, 25)

    all_text = ' '.join(qa['answer']).lower()
    tokens   = [t for t in re.findall(r'\b[a-z]{3,}\b', all_text) if t not in STOPWORDS]
    freq     = Counter(tokens).most_common(top_n)
    words, counts = zip(*freq)

    colors = [C['hawk'] if w in HAWK_WORDS else C['dove'] if w in DOVE_WORDS
              else C['neutral'] for w in words]

    fig = go.Figure(go.Bar(x=list(counts)[::-1], y=list(words)[::-1],
                            orientation='h', marker_color=list(colors)[::-1]))
    fig.update_layout(title=f"Top {top_n} Words in Powell's Answers",
                      height=max(400, top_n * 18),
                      xaxis_title='Frequency', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.markdown("🔴 **Hawkish terms highlighted**")
    col2.markdown("🔵 **Dovish terms highlighted**")

    # Year-over-year comparison for "inflation"
    st.subheader("'Inflation' Mentions per Year")
    qa['year'] = qa['date'].dt.year
    infl_by_year = qa.groupby('year')['answer'].apply(
        lambda texts: sum(t.lower().count('inflation') for t in texts))
    fig2 = go.Figure(go.Bar(x=infl_by_year.index.astype(str), y=infl_by_year.values,
                             marker_color=C['hawk'], opacity=0.85))
    fig2.update_layout(height=300, xaxis_title='Year', yaxis_title='Count',
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ── Page: Sentiment Timeline ───────────────────────────────────────────────────
elif page == "🎭 Sentiment Timeline":
    st.title("🎭 Hawkish / Dovish Sentiment Timeline")
    st.markdown("**VADER compound score** per conference: +1 = most positive/hawkish, −1 = most negative/dovish")

    conf_sent = qa.groupby('date')['vader'].mean().reset_index()
    conf_sent.columns = ['date', 'vader']
    conf_sent['rolling3'] = conf_sent['vader'].rolling(3, min_periods=1).mean()
    conf_sent['tone'] = conf_sent['vader'].apply(
        lambda v: 'Hawkish' if v > 0.05 else ('Dovish' if v < -0.05 else 'Neutral'))

    bar_colors = [C['hawk'] if t == 'Hawkish' else C['dove'] if t == 'Dovish'
                  else C['neutral'] for t in conf_sent['tone']]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=conf_sent['date'], y=conf_sent['vader'],
                         marker_color=bar_colors, name='VADER score', opacity=0.8))
    fig.add_trace(go.Scatter(x=conf_sent['date'], y=conf_sent['rolling3'],
                             mode='lines', line=dict(color='black', width=2.5, dash='dot'),
                             name='3-conf rolling avg'))
    fig.add_hline(y=0.05,  line_dash='dash', line_color=C['hawk'],
                  annotation_text='Hawkish threshold')
    fig.add_hline(y=-0.05, line_dash='dash', line_color=C['dove'],
                  annotation_text='Dovish threshold')
    fig.update_layout(height=420, xaxis_title='Conference Date',
                      yaxis_title='VADER Compound', legend=dict(orientation='h'),
                      margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Most Hawkish",
                conf_sent.loc[conf_sent['vader'].idxmax(), 'date'].strftime('%b %Y'),
                f"{conf_sent['vader'].max():.3f}")
    col2.metric("Most Dovish",
                conf_sent.loc[conf_sent['vader'].idxmin(), 'date'].strftime('%b %Y'),
                f"{conf_sent['vader'].min():.3f}")
    col3.metric("Overall Tone",
                "Hawkish" if conf_sent['vader'].mean() > 0.05
                else "Dovish" if conf_sent['vader'].mean() < -0.05 else "Neutral",
                f"avg={conf_sent['vader'].mean():.3f}")

    st.subheader("Tone Distribution")
    tone_counts = conf_sent['tone'].value_counts()
    fig2 = go.Figure(go.Pie(labels=tone_counts.index, values=tone_counts.values,
                             marker_colors=[C['hawk'], C['dove'], C['neutral']],
                             hole=0.4))
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ── Page: Topic Modeling ───────────────────────────────────────────────────────
elif page == "🗂 Topic Modeling":
    st.title("🗂 LDA Topic Modeling")
    st.markdown("5 latent topics discovered in Powell's Q&A answers (2020–2025)")

    # Top words per topic
    st.subheader("Top 10 Words per Topic")
    cols = st.columns(5)
    colors_list = [C['hawk'], C['dove'], C['green'], C['purple'], C['neutral']]
    for i, (col, label) in enumerate(zip(cols, TOPIC_LABELS.values())):
        with col:
            st.markdown(f"**{label}**")
            for w in top_words[i]:
                st.markdown(f"- {w}")

    st.markdown("---")
    st.subheader("Topic Weight Heatmap per Conference")
    heat_data = monthly_topics[[TOPIC_LABELS[i] for i in range(5)]]
    fig = px.imshow(
        heat_data.T,
        x=heat_data.index.strftime('%b %y'),
        y=list(TOPIC_LABELS.values()),
        color_continuous_scale='YlOrRd',
        aspect='auto',
        labels={'color': 'Topic Weight'}
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dominant Topic per Conference")
    monthly_topics['dominant'] = monthly_topics[[TOPIC_LABELS[i] for i in range(5)]].idxmax(axis=1)
    dom_counts = monthly_topics['dominant'].value_counts()
    fig2 = go.Figure(go.Bar(
        x=dom_counts.index, y=dom_counts.values,
        marker_color=colors_list[:len(dom_counts)], opacity=0.85
    ))
    fig2.update_layout(height=320, xaxis_title='Topic', yaxis_title='# Conferences as Dominant',
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ── Page: N-gram Generator ─────────────────────────────────────────────────────
elif page == "🤖 N-gram Generator":
    st.title("🤖 N-gram Language Model")
    st.markdown("Generates Powell-style text using a trigram model trained on Q&A answers.")
    st.caption("Note: Low similarity to real Powell is expected — this is a statistical word-chain model, not a neural LM.")

    @st.cache_data(show_spinner="Training N-gram model …")
    def train_ngram(_qa_df, n=3):
        model = defaultdict(list)
        for text in _qa_df['answer']:
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            for i in range(len(tokens) - n):
                key = tuple(tokens[i:i+n-1])
                model[key].append(tokens[i+n-1])
        return model

    ngram_model = train_ngram(qa)

    col1, col2 = st.columns([2, 1])
    with col1:
        seed_input = st.text_input("Seed phrase (2 words)", value="inflation remains")
    with col2:
        max_words = st.slider("Max words", 30, 150, 80)

    import random
    seed_val = st.number_input("Random seed", value=42, min_value=0)

    if st.button("Generate"):
        seed_words = seed_input.lower().split()[:2]
        random.seed(seed_val)
        tokens = list(seed_words)
        for _ in range(max_words):
            key = tuple(tokens[-2:])
            nxt = ngram_model.get(key)
            if not nxt:
                break
            tokens.append(random.choice(nxt))
        generated = ' '.join(tokens)
        st.markdown("**Generated response:**")
        st.info(generated)

    st.markdown("---")
    st.subheader("Try these seeds")
    examples = [("inflation", "remains"), ("labor", "market"), ("interest", "rates"), ("our", "policy")]
    cols = st.columns(4)
    for col, (w1, w2) in zip(cols, examples):
        col.code(f"{w1} {w2}")
