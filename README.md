# Emotion-detection-App

A web-based application for emotion classification using a fine-tuned DistilBERT model. The app allows users to input text and returns the predicted emotion with a confidence score and emoji.

## 🚀 Features

- Text-based emotion classification using a BERT model
- Supports real-time input or batch predictions via `.csv` or `.txt` file upload
- Beautiful interface powered by Streamlit
- Prediction explanation with emojis and probability bar charts
- Uses `label_encoder.pkl` to decode predictions into emotion labels

## 🏗️ Project Structure

Emotion-detection-App/
│
├── bert-emotion-model/ # Optional: local model files (can also be downloaded from Hugging Face)
├── src/
│ ├── streamlit_app.py # Main Streamlit interface
│ └── label_encoder.pkl # Pre-trained label encoder for decoding prediction classes
│
├── requirements.txt # All Python dependencies
├── Dockerfile # Docker build instructions
└── README.md # This file
