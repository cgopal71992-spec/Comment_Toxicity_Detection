import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/toxicity_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

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
    """Clean input comment text."""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Prediction Function
# -----------------------------

def predict_toxicity(comment):
    """Predict whether a comment is toxic."""

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

    if probability >= THRESHOLD:
        result = "Toxic"
    else:
        result = "Non-Toxic"

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
    "Enter a comment below to check whether it is toxic or non-toxic."
)


# -----------------------------
# Single Comment Prediction
# -----------------------------

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