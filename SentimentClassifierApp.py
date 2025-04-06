#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Amazon Books Review Sentiment Classifier
------------------------------------------
This app uses a pre-trained sentiment classifier and TF-IDF vectorizer 
(from './models') to classify Amazon Books reviews as Positive or Negative.

Required files:
    - best_sentiment_classifier.pkl
    - vectorizer.pkl
"""

import os
import pickle
import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="Amazon Books Review Sentiment Classifier", layout="centered")

# Custom CSS for a clean appearance
st.markdown("""
<style>
body { background-color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
.stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; font-size: 16px; border-radius: 8px; border: none; transition: background-color 0.3s ease; }
.stButton>button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = 'best_sentiment_classifier.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

@st.cache_resource
def load_resources(model_path: str, vectorizer_path: str):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files missing.")
        return None, None

    with open(model_path, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(vectorizer_path, 'rb') as v_file:
        vectorizer = pickle.load(v_file)
    return model, vectorizer

def classify_review(review: str, model, vectorizer):
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)
    return prediction[0]

st.title("Amazon Books Review Sentiment Classifier")
st.markdown("Enter a review to see if it's **Positive** or **Negative**.")

model, vectorizer = load_resources(MODEL_PATH, VECTORIZER_PATH)
if model is None or vectorizer is None:
    st.stop()

review_text = st.text_area("Your Review:", height=150)

if st.button("Classify"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        sentiment = classify_review(review_text, model, vectorizer)
        st.success(f"Sentiment: **{sentiment}**")
