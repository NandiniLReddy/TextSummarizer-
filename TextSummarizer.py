import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier


@st.cache(allow_output_mutation=True)
def setup_model():
    # Load data
    # Assuming your data is in CSV format
    df = pd.read_csv(
        '/Users/nandinilreddy/Desktop/streamlit/test.csv', encoding="latin1")

    # Feature extraction
    def extract_features(article, highlights):
        sentences = sent_tokenize(article)
        features = []
        labels = []
        for i, sentence in enumerate(sentences):
            features.append({
                'length': len(word_tokenize(sentence)),
                'position': i / len(sentences),
                'named_entity_count': len(list(nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=True).subtrees(filter=lambda t: t.label() == 'NE'))),
                'tf_idf': tfidf.transform([sentence]).mean()
            })
            labels.append(int(sentence in highlights))
        return pd.DataFrame(features), labels

    # Initialize TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(df['article'])

    # Extract features and labels
    df = df.head(1000)

    # Extract features and labels
    features = pd.DataFrame()
    labels = []

    for _, row in df.iterrows():
        article_features, article_labels = extract_features(
            row['article'], row['highlights'])
        features = pd.concat([features, article_features])
        labels.extend(article_labels)

    labels = np.array(labels)

    # Convert lists to arrays for easier manipulation
    features = np.array(features)
    labels = np.array(labels)

    # Split into training and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Train model
    ebm = ExplainableBoostingClassifier()
    ebm.fit(features_train, labels_train)

    return ebm, tfidf


def summarize(ebm, tfidf, article):
    sentences = sent_tokenize(article)
    features = []
    for i, sentence in enumerate(sentences):
        features.append({
            'length': len(word_tokenize(sentence)),
            'position': i / len(sentences),
            'named_entity_count': len(list(nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=True).subtrees(filter=lambda t: t.label() == 'NE'))),
            'tf_idf': tfidf.transform([sentence]).mean()
        })
    features_df = pd.DataFrame(features)
    predictions = ebm.predict_proba(features_df)
    # Get indices of sentences for top 3 predictions
    top_sentence_indices = np.argsort(predictions[:, 1])[-3:]
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    summary = ' '.join(summary_sentences)
    return summary


ebm, tfidf = setup_model()

st.title("Article Summarizer")

article = st.text_area("Enter an article to summarize")

if st.button("Summarize"):
    summary = summarize(ebm, tfidf, article)
    st.text(summary)
