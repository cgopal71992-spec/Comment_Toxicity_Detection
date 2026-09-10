import streamlit as st
import tensorflow as tf
import pickle
import re
import pandas as pd

from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = hf_hub_download(
    repo_id="gopal71992/toxicity-model",
    filename="toxicity_model_v2.keras"
)

TOKENIZER_PATH = "models/tokenizer_v2.pkl"
MAX_LEN = 200
THRESHOLD = 0.70


# -----------------------------
# Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Single Comment Prediction
# -----------------------------
def predict_toxicity(comment):
    cleaned_comment = clean_text(comment)

    sequence = tokenizer.texts_to_sequences([cleaned_comment])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    result = "Toxic" if probability >= THRESHOLD else "Non-Toxic"

    return result, float(probability)


# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="Comment Toxicity Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Comment Toxicity Detection")

st.write(
    "Detect whether a comment is Toxic or Non-Toxic "
    "using a Deep Learning Bi-LSTM model."
)


# ==================================================
# Single Comment Prediction
# ==================================================

st.header("📝 Single Comment Prediction")

comment = st.text_area(
    "Enter your comment:",
    placeholder="Type your comment here..."
)

if st.button("🔍 Predict Toxicity"):

    if comment.strip() == "":
        st.warning("Please enter a comment.")

    else:
        result, probability = predict_toxicity(comment)

        st.subheader("Prediction Result")

        if result == "Toxic":
            st.error("🔴 Toxic Comment")
        else:
            st.success("🟢 Non-Toxic Comment")

        st.write(
            f"**Toxicity Probability:** {probability:.2%}"
        )

        st.progress(probability)


# ==================================================
# Bulk CSV Prediction
# ==================================================

st.header("📁 Bulk CSV Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing a 'comment_text' column",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "comment_text" not in df.columns:

        st.error(
            "CSV must contain a 'comment_text' column."
        )

    else:

        st.success(
            f"File uploaded successfully: {len(df)} comments"
        )

        if st.button("🚀 Predict CSV"):

            predictions = []
            probabilities = []

            for text in df["comment_text"]:

                result, probability = predict_toxicity(text)

                predictions.append(result)
                probabilities.append(probability)

            df["prediction"] = predictions
            df["toxicity_probability"] = probabilities

            st.subheader("📊 Prediction Results")

            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="toxicity_predictions.csv",
                mime="text/csv"
            )


# ==================================================
# Model Information
# ==================================================

st.header("ℹ️ Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Bi-LSTM")

with col2:
    st.metric("F1 Score", "78.56%")

with col3:
    st.metric("Threshold", "0.70")

st.caption(
    "Model trained using the Jigsaw Toxic Comment dataset."
)
