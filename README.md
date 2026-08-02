# Sentiment Analysis System

A clean, interactive, and professional Streamlit web application designed for classifying IMDb Movie Reviews. This application leverages a custom-built Natural Language Processing (NLP) pipeline and machine learning classifier.

## Project Structure

```
Sentiment-Analysis/
│
├── app.py              # Streamlit Web App Interface
├── models.py           # Custom pure-Python/NumPy ML model classes
├── model.pkl           # Trained Logistic Regression weights (serialized)
├── vectorizer.pkl      # Fitted TF-IDF Vectorizer parameters (serialized)
├── requirements.txt    # Application dependencies
└── README.md           # Instructions and documentation
```

## Technical Features

### Custom NumPy Architecture
To ensure high portability, bypass OS DLL loading blocks (like Windows Application Control Policies), and eliminate unnecessary third-party package dependencies during inference, the machine learning models are implemented in pure Python and `numpy` (found in `models.py`):
1. **SimpleTfidfVectorizer**: Implements regular expression cleaning, lowercase normalization, URL/HTML tag removal, basic English suffix stemming, stopword filtering, TF-IDF calculation, and L2 vector normalization.
2. **SimpleLogisticRegression**: Implements a standard weights-and-bias model with sigmoid activation, supporting `predict_proba` for probability distributions.
3. **Interactive Report Cards**: Supports inputting an optional Movie Title to compile personalized, styled sentiment reports featuring the review and its confidence levels.

This design enables running the model inside the web app **without requiring heavy libraries like scikit-learn, scipy, or NLTK** for prediction, making the application extremely lightweight and fast.

## How to Run Locally

### 1. Prerequisites
- **64-bit Python 3.7+**: Streamlit depends on `pyarrow`, which does not support 32-bit Windows. You must use a 64-bit installation of Python.
- **Python Launcher (`py`)**: If `python` or `pip` are not in your system's PATH, you can use the Windows Python launcher `py`.

### 2. Install Dependencies
Navigate to the `Sentiment-Analysis` directory and run:
```bash
py -m pip install -r requirements.txt
```
*(If Python is in your PATH, you can run `pip install -r requirements.txt`)*

### 3. Run the Application
Execute the following command to launch the Streamlit server:
```bash
py -m streamlit run app.py
```
*(If Python/Scripts are in your PATH, you can run `streamlit run app.py`)*

This will open the application in your default web browser (typically at `http://localhost:8501`).

## How to Deploy to Streamlit Cloud

1. Create a public repository on GitHub containing the contents of the `Sentiment-Analysis/` folder at the root.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch, and specify `app.py` as the main file path.
4. Click **Deploy!** Streamlit will automatically install the packages listed in `requirements.txt` and launch your live application.
