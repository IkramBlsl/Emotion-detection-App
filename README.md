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

All preprocessing, fine-tuning, and evaluation were done using the `transformers` and `datasets` libraries.

The model and tokenizer are hosted on Hugging Face Hub:  
🔗 [`Ikraaaam/model_emotion_bert`](https://huggingface.co/Ikraaaam/model_emotion_bert)

## ⚙️ Technologies

- `transformers`
- `torch`
- `streamlit`
- `pandas`, `numpy`
- `scikit-learn` (for label encoding)


## 🔎 Other Models Explored

Before finalizing the BERT-based classifier, several machine learning and deep learning models were tested:

- **XGBoost Classifier**  
  Tree-based gradient boosting trained on TF-IDF features. Good baseline but lacked semantic understanding.

- **Logistic Regression**  
  Used as a fast, interpretable baseline. Trained on bag-of-words and TF-IDF vectors. Performance was limited.

- **Recurrent Neural Network (RNN)**  
  Built with PyTorch using GRU units. Performed reasonably but required more tuning and training time.

These models helped us validate the superiority of transformers in emotion classification.


## 📊 Results

| Model              | Accuracy  | Notes                            |
|-------------------|----------|----------------------------------|
| **DistilBERT**     | *0.94%*  | Best overall performance         |
| XGBoost            | *0.74*  | Limited on contextual features   |
| Logistic Regression | *0.87*  | Fast but less expressive         |
| RNN (GRU)         | *0.35*  | Decent but underperformed BERT   |


✨ Demo

Try the app directly on Hugging Face Spaces:
🔗 Launch Emotion Detection App
