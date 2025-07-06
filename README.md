# Emotion-detection-App

A web-based application for emotion classification using a fine-tuned DistilBERT model. The app allows users to input text and returns the predicted emotion with a confidence score and emoji.

## 🚀 Features

- Text-based emotion classification using a BERT model
- Supports real-time input or batch predictions via `.csv` or `.txt` file upload
- Beautiful interface powered by Streamlit
- Prediction explanation with emojis and probability bar charts
- Uses `label_encoder.pkl` to decode predictions into emotion labels

## Install dependencies

pip install -r requirements.txt


## Run the App localy

streamlit run streamlit_app.py

## Model & Inference

The model used is a fine-tuned DistilBERT classifier, hosted publicly on Hugging Face under the repo:
Ikraaaam/model_emotion_bert

It predicts one of the following emotions:

    Joy 😄

    Sadness 😞

    Anger 😡

    Fear 😨

    Love ❤️

    Surprise 😲



## File Upload Format

    .txt file: one sentence per line

    .csv file: must contain a column named text

Example .csv format:

    I am so happy today!
  
    This is terrible...
  
    I'm in love with this song!

