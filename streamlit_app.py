import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from wordcloud import WordCloud
from io import BytesIO
import base64

# ✅ Ensure VADER lexicon is available before using SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

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

st.title('📰 Sentiment Analysis Dashboard')
st.markdown('Analyze sentiment of news headlines using NewsAPI and visualize results.')

with st.sidebar:
    st.header("Configuration")
    apiKey = "81d72cce2e69445fbd4da2d4ad7ec5d0"  # Hardcoded NewsAPI key
    search_query = st.text_input('Enter search query (topic, keyword, etc.):', value='')
    selected_model = st.selectbox('Sentiment Model', ['VADER', 'TextBlob'])

if apiKey and search_query:
    url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={apiKey}'
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('status') != 'ok':
            st.error(f"Error fetching news: {data.get('message', 'Unknown error')}")
            st.stop()
        articles = data.get('articles', [])
        if not articles:
            st.warning("No articles found for your query.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        st.stop()

    df = pd.DataFrame(articles)
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    for text in df['text']:
        if selected_model.lower() == 'vader':
            score = analyzer.polarity_scores(text)
            compound = score['compound']
            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        elif selected_model.lower() == 'textblob':
            from textblob import TextBlob
            tb = TextBlob(text)
            polarity = tb.sentiment.polarity
            if polarity > 0.05:
                sentiment = 'Positive'
            elif polarity < -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        else:
            sentiment = 'Neutral'
        sentiments.append(sentiment)
    df['sentiment'] = sentiments

    total_texts = len(df)
    positive = (df['sentiment'] == 'Positive').sum()
    negative = (df['sentiment'] == 'Negative').sum()
    neutral = (df['sentiment'] == 'Neutral').sum()
    overall_score = int(100 * positive / total_texts) if total_texts > 0 else 0

    # Layout: Metrics and Pie Chart
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Total Texts Analyzed', total_texts)
        st.metric('Overall Sentiment Score (% Positive)', overall_score)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader('Sentiment Breakdown')
        pie_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [positive, negative, neutral]
        })
        fig_pie = px.pie(pie_df, names='Sentiment', values='Count', color='Sentiment',
                         color_discrete_map={'Positive':'green','Negative':'red','Neutral':'gray'})
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent mentions
    st.subheader('Recent Mentions')
    st.dataframe(df[['text', 'sentiment']].head(10).style.applymap(
        lambda v: 'background-color: #d4edda' if v=='Positive' else ('background-color: #f8d7da' if v=='Negative' else 'background-color: #e2e3e5'), subset=['sentiment']))

    # Time-Series Analysis (if publishedAt available)
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        ts_df = df.dropna(subset=['publishedAt'])
        ts_df['date'] = ts_df['publishedAt'].dt.date
        ts_sentiment = ts_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        st.subheader('Sentiment Over Time')
        fig_ts = px.line(ts_sentiment, x=ts_sentiment.index, y=['Positive','Negative','Neutral'], markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Word Clouds
    def generate_wordcloud(texts):
        wc = WordCloud(width=400, height=200, background_color='white').generate(' '.join(texts))
        img = BytesIO()
        wc.to_image().save(img, format='PNG')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    pos_texts = df[df['sentiment'] == 'Positive']['text'].tolist()
    neg_texts = df[df['sentiment'] == 'Negative']['text'].tolist()
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
        st.subheader('Word Cloud: Positive Texts')
        if pos_texts:
            pos_wc = generate_wordcloud(pos_texts)
            st.image(f'data:image/png;base64,{pos_wc}')
        else:
            st.write('No positive texts found.')
        st.markdown('</div>', unsafe_allow_html=True)
    with wc_col2:
        st.markdown('<div class="wordcloud-card">', unsafe_allow_html=True)
        st.subheader('Word Cloud: Negative Texts')
        if neg_texts:
            neg_wc = generate_wordcloud(neg_texts)
            st.image(f'data:image/png;base64,{neg_wc}')
        else:
            st.write('No negative texts found.')
        st.markdown('</div>', unsafe_allow_html=True)
