import streamlit as st
import os
import sys
import pickle
import pandas as pd
import numpy as np
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models import SimpleTfidfVectorizer, SimpleLogisticRegression

# Set up page configuration
st.set_page_config(
    page_title="IMDb Sentiment Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR PANEL ---
st.sidebar.markdown("# 🎬 Sentiment Analytics")
st.sidebar.markdown("---")

# Page Navigation in Sidebar
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    [
        "🔮 Single Review Prediction", 
        "📂 Batch Analysis (CSV Upload)", 
        "📊 Project Insights & EDA"
    ]
)
st.sidebar.markdown("---")

st.sidebar.info(
    "**IMDb Movie Reviews Analytics Dashboard**\n\n"
    "Classify raw English movie reviews into Positive or Negative categories in real-time or in batch."
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

# --- MAIN PAGE HEADER ---
st.title("🎬 IMDb Movie Review Sentiment Analytics")
st.markdown("##### An interactive dashboard for real-time inference, batch predictions, and model performance metrics.")

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

# Render selected page from sidebar navigation
if models_loaded:
    # ----------------------------------------------------
    # PAGE 1: SINGLE REVIEW PREDICTION
    # ----------------------------------------------------
    if page == "🔮 Single Review Prediction":
        st.markdown("### ✍️ Analyze a Single Movie Review")
        st.write("Input a movie title and copy-paste a review below to inspect its predicted sentiment classification.")
        
        movie_title = st.text_input(
            "🎬 Movie Title (Optional):",
            placeholder="E.g., Inception, The Dark Knight, Titanic",
            key="single_movie_title"
        )
        user_input = st.text_area(
            "Type or paste your review below:",
            placeholder="E.g., This movie was absolute perfection. The direction and cast did an outstanding job!",
            height=120,
            key="single_review_input"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            predict_clicked = st.button("🔮 Predict Sentiment", use_container_width=True)

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
                    
                    st.markdown("**Review Analysed:**")
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

    # ----------------------------------------------------
    # PAGE 2: BATCH ANALYSIS (CSV UPLOAD)
    # ----------------------------------------------------
    elif page == "📂 Batch Analysis (CSV Upload)":
        st.markdown("### 📂 Batch Review Analysis via CSV Upload")
        st.write("Upload a CSV file containing multiple reviews to process them in batch, view charts, and download predictions.")
        
        # Format guidelines alert box
        st.info(
            "👉 **CSV Format Guide:** The uploaded file must contain a column named exactly **`review`** or **`text`** containing the reviews. "
            "It can optionally contain a **`movie`** or **`title`** column."
        )
        
        # Download Sample Template
        sample_df = pd.DataFrame({
            "movie": ["Inception", "The Last Airbender", "The Dark Knight", "Avatar 2"],
            "review": [
                "Absolutely mind-bending and spectacular cinematography! A masterpiece.",
                "Terrible screenplay, bad acting, and a complete waste of time.",
                "Heath Ledger's performance was legendary. Brilliant action sequences.",
                "Visually stunning but the plot felt extremely repetitive and boring."
            ]
        })
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_movie_reviews.csv",
            mime="text/csv"
        )
        
        st.write("---")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Verify required column exists
                target_col = None
                for col in ["review", "text"]:
                    if col in df.columns:
                        target_col = col
                        break
                
                if target_col is None:
                    st.error("❌ **Error:** Could not find a column named `review` or `text` in your CSV. Please check the column headers.")
                else:
                    st.write(f"Detected target review column: **`{target_col}`**")
                    
                    with st.spinner("Processing batch predictions..."):
                        reviews_list = df[target_col].fillna("").astype(str).tolist()
                        
                        predictions = []
                        pos_probabilities = []
                        neg_probabilities = []
                        
                        for rev in reviews_list:
                            # Preprocess and vectorize
                            tokens = vectorizer.clean_and_tokenize(rev)
                            cleaned_text = " ".join(tokens)
                            vec = vectorizer.transform([cleaned_text])
                            
                            # Inference
                            probs = model.predict_proba(vec)[0]
                            prob_neg, prob_pos = probs
                            
                            pos_probabilities.append(prob_pos)
                            neg_probabilities.append(prob_neg)
                            
                            if prob_pos >= 0.50:
                                predictions.append("Positive")
                            else:
                                predictions.append("Negative")
                        
                        # Add prediction results to the DataFrame
                        df["Predicted Sentiment"] = predictions
                        df["Positive Probability (%)"] = [round(p * 100, 2) for p in pos_probabilities]
                        df["Negative Probability (%)"] = [round(p * 100, 2) for p in neg_probabilities]
                        
                        # Summary KPIs
                        total_count = len(df)
                        pos_count = sum(1 for p in predictions if p == "Positive")
                        neg_count = total_count - pos_count
                        pos_ratio = (pos_count / total_count) * 100
                        
                        st.markdown("### 📊 Batch Execution Summary")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Reviews Processed", f"{total_count:,}")
                        with m2:
                            st.metric("🟢 Positive Sentiments", f"{pos_count:,}")
                        with m3:
                            st.metric("🔴 Negative Sentiments", f"{neg_count:,}")
                        with m4:
                            st.metric("Positive Sentiment Ratio", f"{pos_ratio:.2f}%")
                            
                        # Chart
                        st.markdown("#### Sentiment Distribution")
                        chart_data = pd.DataFrame({
                            "Sentiment": ["Positive", "Negative"],
                            "Count": [pos_count, neg_count]
                        }).set_index("Sentiment")
                        st.bar_chart(chart_data)
                        
                        # Download Predicted Results CSV Button
                        out_buffer = io.BytesIO()
                        df.to_csv(out_buffer, index=False)
                        st.download_button(
                            label="📥 Download Predicted Results (CSV)",
                            data=out_buffer.getvalue(),
                            file_name="sentiment_predictions_output.csv",
                            mime="text/csv"
                        )
                        
                        # Dataframe Preview
                        st.markdown("#### Preview of Processed Records (First 100 rows)")
                        st.dataframe(df.head(100), use_container_width=True)
                        
            except Exception as e:
                st.error(f"Failed to process CSV file: {e}")

    # ----------------------------------------------------
    # PAGE 3: PROJECT INSIGHTS & EDA
    # ----------------------------------------------------
    elif page == "📊 Project Insights & EDA":
        st.markdown("### 📊 Project Insights & Model Performance")
        st.write("Review the model evaluation metrics, accuracy reports, and exploratory data analysis details from the training pipeline.")
        
        # Model Comparison Metrics Table
        st.markdown("#### 1. Machine Learning Model Comparison")
        st.write("During the training and evaluation phase, three different classification algorithms were trained and validated on a 20% stratified test split from the IMDb dataset:")
        
        comparison_df = pd.DataFrame({
            "Model Name": ["Logistic Regression", "Multinomial Naive Bayes", "Linear Support Vector Machine (SVC)"],
            "Accuracy (%)": ["82.42%", "81.65%", "81.90%"],
            "Precision (%)": ["82.26%", "83.69%", "81.59%"],
            "Recall (%)": ["82.72%", "78.61%", "82.46%"],
            "F1-Score (%)": ["82.49%", "81.07%", "82.02%"]
        })
        st.table(comparison_df)
        
        st.info(
            "💡 **Design Decision:** The custom Logistic Regression model was selected for the live web application deployment. "
            "It yields the highest overall Accuracy (82.42%) and F1-Score (82.49%), and is highly efficient for real-time predictions."
        )
        
        # Text Preprocessing flowchart
        st.markdown("#### 2. Text Preprocessing & TF-IDF Extraction Pipeline")
        st.write("All raw reviews pass through a clean-and-tokenize function before classification. This ensures that punctuation, formatting, and high-frequency terms do not skew the sentiment score:")
        
        flowchart = """
        [Raw Review Text]
               │
               ▼
        [Lowercasing] (Standardizes character casing)
               │
               ▼
        [HTML & URL Removal] (Regex filters out `<br />` and `http://` tags)
               │
               ▼
        [Punctuation & Number Removal] (Filters out non-alphabetical characters)
               │
               ▼
        [Stopword Filtering] (Excludes common words like 'the', 'a', 'is')
               │
               ▼
        [Stemming/Lemmatization] (Reduces words to base form, e.g., 'outstandingly' -> 'outstand')
               │
               ▼
        [TF-IDF Vectorization] (Extracts numerical counts of 3,000 top n-grams)
               │
               ▼
        [Model Classification] (Logistic regression predicts positive/negative probability)
        """
        st.code(flowchart, language="text")
        
        st.markdown("#### 3. IMDb Dataset Statistics")
        st.write("The underlying IMDb sentiment classification dataset consists of **50,000 highly polarized movie reviews**:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Total Records", "50,000")
            st.metric("Positive Samples", "25,000 (50.0%)")
        with col_c2:
            st.metric("Negative Samples", "25,000 (50.0%)")
            st.metric("Max Feature Dimensions", "3,000 N-grams")
