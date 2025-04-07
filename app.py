
"""
Amazon Books Review Sentiment Classifier 
--------------------------------------------------------------------
This app uses a pre-trained sentiment classifier and TF-IDF vectorizer 
 to classify Amazon Books reviews as Positive or Negative.
Users can choose between manual review entry or batch CSV upload via the sidebar.
"""

import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px 


st.set_page_config(page_title="Amazon Books Review Sentiment Classifier", layout="centered")


st.markdown("""
<style>
body { background-color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
.stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; font-size: 16px; border-radius: 8px; border: none; transition: background-color 0.3s ease; }
.stButton>button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

# Model and Vectorizer paths
MODEL_PATH = 'best_sentiment_classifier.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

@st.cache_resource
def load_resources(model_path: str, vectorizer_path: str):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files missing. Please check the './models' directory.")
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

# Sidebar for mode selection
st.sidebar.title("Choose Analysis Mode")
mode = st.sidebar.radio("Select Mode", ["Enter Review Manually", "Upload CSV for Batch Analysis"])

# Load model and vectorizer resources
model, vectorizer = load_resources(MODEL_PATH, VECTORIZER_PATH)
if model is None or vectorizer is None:
    st.stop()

st.title("Amazon Books Review Sentiment Classifier")
st.markdown("This tool classifies reviews as **Positive** or **Negative**. Choose your mode from the sidebar.")

if mode == "Enter Review Manually":
    st.subheader("Manual Review Input")
    review_text = st.text_area("Enter your review here:", height=150)
    if st.button("Classify Review"):
        if not review_text.strip():
            st.warning("Please enter a review before classifying.")
        else:
            sentiment = classify_review(review_text, model, vectorizer)
            st.success(f"Predicted Sentiment: **{sentiment}**")

elif mode == "Upload CSV for Batch Analysis":
    st.subheader("Batch Analysis via CSV Upload")
    st.markdown("Ensure your CSV file contains a column named **'review'**.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'review' not in data.columns:
                st.error("The CSV file must contain a 'review' column.")
            else:
                results = []
                for idx, row in data.iterrows():
                    review = row['review']
                    sentiment = classify_review(review, model, vectorizer)
                    results.append({
                        "Review": review,
                        "Predicted Sentiment": sentiment
                    })
                results_df = pd.DataFrame(results)
                
                # Optional data table display
                if st.checkbox("Show raw results table"):
                    st.dataframe(results_df)
                
                # Calculate overall sentiment counts and display
                sentiment_counts = results_df["Predicted Sentiment"].value_counts()
                st.markdown("### Overall Sentiment Counts")
                st.write(sentiment_counts)
                
                # Professional Pie Chart Visualization
                pie_fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Sentiment Proportions",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(pie_fig)
        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")
