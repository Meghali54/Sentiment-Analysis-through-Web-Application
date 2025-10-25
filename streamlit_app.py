import datetime
import pandas as pd
import numpy as np
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO
import base64
import plotly.express as px
import nltk

# Ensure lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# --- Page Config ---
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# --- Custom Styles ---
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.wordcloud-card {
    background: #fff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("📰 Sentiment Analysis Dashboard")
st.markdown("Analyze sentiment of news headlines using NewsAPI or demo data.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your NewsAPI Key (leave blank for demo mode)", type="password")
    query = st.text_input("Search Query (topic/keyword)", value="AI")
    model_choice = st.selectbox("Sentiment Model", ["VADER", "TextBlob"])

# --- Cached Fetch Function ---
@st.cache_data(show_spinner=True)
def fetch_news(query, api_key):
    """Fetch news from NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        return data.get("articles", [])
    except Exception:
        return []

# --- Demo Data Fallback ---
def get_demo_data():
    sample_data = [
        {
            "title": "AI breakthrough revolutionizes healthcare diagnostics",
            "description": "New machine learning models improve disease detection rates dramatically.",
            "publishedAt": "2025-10-20T09:00:00Z"
        },
        {
            "title": "Tech stocks tumble after AI regulation news",
            "description": "Markets react negatively to new AI restrictions proposed by the EU.",
            "publishedAt": "2025-10-21T11:30:00Z"
        },
        {
            "title": "AI helps create stunning digital artwork",
            "description": "Artists are using generative models to enhance creativity.",
            "publishedAt": "2025-10-22T10:15:00Z"
        },
        {
            "title": "Concerns rise over deepfake technology misuse",
            "description": "Experts warn about growing ethical issues in AI-generated content.",
            "publishedAt": "2025-10-23T12:45:00Z"
        },
        {
            "title": "AI-powered language models boost education accessibility",
            "description": "Schools adopt AI tutors to improve learning outcomes.",
            "publishedAt": "2025-10-24T08:00:00Z"
        },
    ]
    return pd.DataFrame(sample_data)

# --- Get Articles ---
if api_key and query:
    articles = fetch_news(query, api_key)
    if not articles:
        st.warning("⚠️ No articles found or invalid API key. Showing demo data instead.")
        df = get_demo_data()
    else:
        df = pd.DataFrame(articles)
else:
    st.info("🔹 Using demo mode — enter your NewsAPI key to fetch live articles.")
    df = get_demo_data()

# --- Combine text fields ---
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# --- Cached Sentiment Analysis ---
@st.cache_data(show_spinner=False)
def analyze_sentiment(texts, model):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    if model.lower() == "vader":
        for t in texts:
            score = analyzer.polarity_scores(t)["compound"]
            if score >= 0.05:
                sentiments.append("Positive")
            elif score <= -0.05:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")
    else:
        from textblob import TextBlob
        for t in texts:
            polarity = TextBlob(t).sentiment.polarity
            if polarity > 0.05:
                sentiments.append("Positive")
            elif polarity < -0.05:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")
    return sentiments

# --- Run Sentiment ---
df["sentiment"] = analyze_sentiment(df["text"], model_choice)

# --- Metrics ---
total = len(df)
pos = (df["sentiment"] == "Positive").sum()
neg = (df["sentiment"] == "Negative").sum()
neu = (df["sentiment"] == "Neutral").sum()
score = int(100 * pos / total) if total > 0 else 0

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Articles", total)
    st.metric("Overall Sentiment (% Positive)", score)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Sentiment Breakdown")
    pie_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [pos, neg, neu]
    })
    fig_pie = px.pie(
        pie_df, names="Sentiment", values="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Recent Mentions ---
st.subheader("Recent Mentions")
st.dataframe(
    df[["text", "sentiment"]].head(10).style.applymap(
        lambda v: (
            "background-color: #d4edda" if v == "Positive"
            else "background-color: #f8d7da" if v == "Negative"
            else "background-color: #e2e3e5"
        ),
        subset=["sentiment"]
    )
)

# --- Sentiment Over Time ---
if "publishedAt" in df.columns:
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    ts_df = df.dropna(subset=["publishedAt"])
    ts_df["date"] = ts_df["publishedAt"].dt.date
    ts_data = ts_df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    st.subheader("Sentiment Over Time")
    fig_ts = px.line(ts_data, x=ts_data.index, y=["Positive", "Negative", "Neutral"], markers=True)
    st.plotly_chart(fig_ts, use_container_width=True)

# --- Word Clouds ---
@st.cache_data(show_spinner=False)
def generate_wordcloud(texts):
    from wordcloud import WordCloud
    if not texts:
        return None
    wc = WordCloud(width=400, height=200, background_color="white").generate(" ".join(texts))
    img = BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

st.subheader("Word Clouds")
colA, colB = st.columns(2)
with colA:
    st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
    st.subheader("Positive Texts")
    pos_wc = generate_wordcloud(df[df["sentiment"] == "Positive"]["text"].tolist())
    if pos_wc:
        st.image(f"data:image/png;base64,{pos_wc}")
    else:
        st.info("No positive texts found.")
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
    st.subheader("Negative Texts")
    neg_wc = generate_wordcloud(df[df["sentiment"] == "Negative"]["text"].tolist())
    if neg_wc:
        st.image(f"data:image/png;base64,{neg_wc}")
    else:
        st.info("No negative texts found.")
    st.markdown("</div>", unsafe_allow_html=True)
