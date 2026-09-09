
# Comment Toxicity Detection with Deep Learning

## Project Overview

This project detects whether an online comment is Toxic or Non-Toxic using a Deep Learning Bi-LSTM model.

The project includes text preprocessing, model training, evaluation, and a Streamlit web application for real-time and bulk CSV predictions.

## Features

- Text preprocessing
- Bi-LSTM based toxicity classification
- Class imbalance handling
- Real-time comment prediction
- Bulk CSV prediction
- Toxicity probability score
- Downloadable prediction results
- Streamlit web interface

## Dataset

The project uses the Jigsaw Toxic Comment dataset.

The original dataset contains six toxicity categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

For this project, these categories are combined into a binary target:

- 0 = Non-Toxic
- 1 = Toxic

## Model Performance

Final Model: Bidirectional LSTM (Bi-LSTM)

- Accuracy: 94.28%
- Precision: 73.61%
- Recall: 84.22%
- F1 Score: 78.56%
- Classification Threshold: 0.70

## Workflow

1. Load dataset
2. Exploratory data analysis
3. Text preprocessing
4. Tokenization
5. Sequence padding
6. Train-validation split
7. Bi-LSTM model training
8. Class imbalance handling
9. Model evaluation
10. Threshold tuning
11. Model saving
12. Streamlit deployment

## Streamlit Application

The application supports:

### Single Comment Prediction

Users can enter a comment and receive:

- Toxic or Non-Toxic prediction
- Toxicity probability

### Bulk CSV Prediction

Users can upload a CSV containing a `comment_text` column.

The application predicts all comments and provides a downloadable CSV file.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- NLP
- Deep Learning
- Bi-LSTM

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Conclusion

This project demonstrates how Deep Learning and Natural Language Processing can be used to identify toxic comments and support online content moderation.
