import streamlit as st
import os
import sys
import pickle
import numpy as np

# Ensure the app folder is in sys.path so pickle can resolve classes from models.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models import SimpleTfidfVectorizer, SimpleLogisticRegression

# Set up page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (No custom HTML style blocks needed - using native Streamlit theme components)

# --- SIDEBAR PANEL ---
st.sidebar.markdown("# 🎬 Project Dashboard")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Title:**\n\n"
    "Sentiment Analysis on IMDb Movie Reviews Using Machine Learning\n\n"
    "**Version:**\n\n"
    "1.0.0 (Internship Demo)\n\n"
    "**Objective:**\n\n"
    "Classify raw English movie reviews into sentiments (Positive, Negative, or Neutral) using a custom Logistic Regression model."
)

st.sidebar.markdown("### 📊 Model Details")
st.sidebar.markdown(
    "- **Model Type:** Custom Logistic Regression\n"
    "- **Features:** TF-IDF (3,000 Unigrams/Bigrams)\n"
    "- **Training Set Size:** 25,000 reviews\n"
    "- **Validation Accuracy:** 82.42%"
)

st.sidebar.markdown("### 📂 Project Structure")
st.sidebar.code(
    "Sentiment-Analysis/\n"
    "├── app.py\n"
    "├── models.py\n"
    "├── model.pkl\n"
    "├── vectorizer.pkl\n"
    "├── requirements.txt\n"
    "└── README.md"
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by **Data Science Intern**")

# --- MAIN PAGE ---
st.title("🎬 Sentiment Analysis System")
st.markdown("##### A clean and interactive interface to test IMDb movie review predictions in real-time.")

# Load model and vectorizer with proper error handling
model_path = os.path.join(current_dir, "model.pkl")
vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

model = None
vectorizer = None
models_loaded = False

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error(
        "🚨 **Error: Pickled model files not found!**\n\n"
        "Please ensure `model.pkl` and `vectorizer.pkl` exist in the application folder.\n\n"
        "If you are running this project for the first time, run the training pipeline first to generate the pickle files."
    )
else:
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        models_loaded = True
    except Exception as e:
        st.error(f"🚨 **Failed to load pickled model files:** {e}")

# Text input and evaluation section
if models_loaded:
    st.markdown("### ✍️ Enter Movie Review Details")
    movie_title = st.text_input(
        "🎬 Movie Title (Optional):",
        placeholder="E.g., Inception, The Dark Knight, Titanic"
    )
    user_input = st.text_area(
        "Type or paste your review below:",
        placeholder="E.g., This movie was absolute perfection. The direction and cast did an outstanding job!",
        height=120
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        predict_clicked = st.button("🔮 Predict", use_container_width=True)

    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review first! The input cannot be empty.")
        else:
            with st.spinner("Analyzing review sentiment..."):
                # 1. Preprocess the text exactly like the training pipeline
                cleaned = vectorizer.clean_and_tokenize(user_input)
                cleaned_text = " ".join(cleaned)
                
                # 2. Transform the text to TF-IDF numerical vectors
                vectorized = vectorizer.transform([cleaned_text])
                
                # 3. Predict sentiment probability
                probs = model.predict_proba(vectorized)
                prob_neg, prob_pos = probs[0]
                
                # 4. Display result (binary classification: Positive or Negative)
                if prob_pos >= 0.50:
                    sentiment = "Positive 😊"
                    display_prob = prob_pos * 100
                else:
                    sentiment = "Negative 😞"
                    display_prob = prob_neg * 100

                if movie_title.strip():
                    st.markdown(f"### 🎬 Sentiment Report for **{movie_title.strip()}**")
                else:
                    st.markdown("### 📊 Sentiment Report")
                
                if sentiment == "Positive 😊":
                    st.success(f"### **Result: {sentiment}** ({display_prob:.2f}% Confidence)")
                else:
                    st.error(f"### **Result: {sentiment}** ({display_prob:.2f}% Confidence)")
                
                st.markdown(f"**Review Analysed:**")
                st.info(f"*{user_input}*")
                
                # Visualizing confidence scores
                st.markdown("#### Probability Distribution")
                st.progress(float(prob_pos))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="🟢 Positive Probability", value=f"{prob_pos * 100:.2f}%")
                with c2:
                    st.metric(label="🔴 Negative Probability", value=f"{prob_neg * 100:.2f}%")
                
                # Display metrics summary details
                with st.expander("🔍 Show Text Preprocessing Details"):
                    st.write(f"**Original review:** {user_input}")
                    st.write(f"**Cleaned tokens:** {cleaned}")
