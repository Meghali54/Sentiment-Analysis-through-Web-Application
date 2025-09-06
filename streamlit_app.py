import datetime
import pandas as pd
import numpy as np
import streamlit as st
import requests
from vaderSentiment import SentimentIntensityAnalyzer  # FIXED IMPORT
import nltk

# Download VADER lexicon only if not present
nltk.download('vader_lexicon', quiet=True)

st.markdown("""
<style>
.metric-card {background: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;}
.wordcloud-card {background: #fff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title('📰 Sentiment Analysis Dashboard')
st.markdown('Analyze sentiment of news headlines using NewsAPI and visualize results.')

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")
    apiKey = "81d72cce2e69445fbd4da2d4ad7ec5d0"
    search_query = st.text_input('Enter search query (topic, keyword, etc.):', value='')
    selected_model = st.selectbox('Sentiment Model', ['VADER', 'TextBlob'])

# ---------------- Fetch News ----------------
@st.cache_data(show_spinner=False)
def fetch_news(query, api_key):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    resp = requests.get(url).json()
    if resp.get('status') != 'ok':
        return []
    return resp.get('articles', [])

if apiKey and search_query:
    articles = fetch_news(search_query, apiKey)
    if not articles:
        st.warning("No articles found for your query.")
        st.stop()

    df = pd.DataFrame(articles)
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # ---------------- Sentiment Analysis ----------------
    @st.cache_data(show_spinner=False)
    def analyze_sentiment(texts, model):
        results = []
        analyzer = SentimentIntensityAnalyzer()
        if model.lower() == 'vader':
            for text in texts:
                score = analyzer.polarity_scores(text)['compound']
                if score >= 0.05: sentiment = 'Positive'
                elif score <= -0.05: sentiment = 'Negative'
                else: sentiment = 'Neutral'
                results.append(sentiment)
        else:
            from textblob import TextBlob  # Import inside function for faster cold start
            for text in texts:
                polarity = TextBlob(text).sentiment.polarity
                if polarity > 0.05: sentiment = 'Positive'
                elif polarity < -0.05: sentiment = 'Negative'
                else: sentiment = 'Neutral'
                results.append(sentiment)
        return results

    df['sentiment'] = analyze_sentiment(df['text'], selected_model)

    total_texts = len(df)
    positive = (df['sentiment'] == 'Positive').sum()
    negative = (df['sentiment'] == 'Negative').sum()
    neutral = (df['sentiment'] == 'Neutral').sum()
    overall_score = int(100 * positive / total_texts) if total_texts > 0 else 0

    # ---------------- Metrics ----------------
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Total Texts Analyzed', total_texts)
        st.metric('Overall Sentiment Score (% Positive)', overall_score)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Pie Chart ----------------
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader('Sentiment Breakdown')
        import plotly.express as px
        pie_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [positive, negative, neutral]
        })
        fig_pie = px.pie(pie_df, names='Sentiment', values='Count', color='Sentiment',
                         color_discrete_map={'Positive':'green','Negative':'red','Neutral':'gray'})
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Recent Mentions ----------------
    st.subheader('Recent Mentions')
    st.dataframe(df[['text','sentiment']].head(10).style.applymap(
        lambda v: 'background-color: #d4edda' if v=='Positive' else ('background-color: #f8d7da' if v=='Negative' else 'background-color: #e2e3e5'),
        subset=['sentiment']))

    # ---------------- Time Series ----------------
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        ts_df = df.dropna(subset=['publishedAt'])
        ts_df['date'] = ts_df['publishedAt'].dt.date
        ts_sentiment = ts_df.groupby(['date','sentiment']).size().unstack(fill_value=0)
        st.subheader('Sentiment Over Time')
        fig_ts = px.line(ts_sentiment, x=ts_sentiment.index, y=['Positive','Negative','Neutral'], markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ---------------- Word Clouds ----------------
    @st.cache_data(show_spinner=False)
    def generate_wordcloud(texts):
        from wordcloud import WordCloud  # Import inside function for faster cold start
        from io import BytesIO
        import base64
        if not texts: return None
        wc = WordCloud(width=400, height=200, background_color='white').generate(' '.join(texts))
        img = BytesIO()
        wc.to_image().save(img, format='PNG')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    pos_texts = df[df['sentiment']=='Positive']['text'].tolist()
    neg_texts = df[df['sentiment']=='Negative']['text'].tolist()
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
        st.subheader('Word Cloud: Positive Texts')
        pos_wc = generate_wordcloud(pos_texts)
        if pos_wc: st.image(f'data:image/png;base64,{pos_wc}')
        else: st.write('No positive texts found.')
        st.markdown('</div>', unsafe_allow_html=True)
    with wc_col2:
        st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
        st.subheader('Word Cloud: Negative Texts')
        neg_wc = generate_wordcloud(neg_texts)
        if neg_wc: st.image(f'data:image/png;base64,{neg_wc}')
        else: st.write('No negative texts found.')
        st.markdown('</div>', unsafe_allow_html=True)