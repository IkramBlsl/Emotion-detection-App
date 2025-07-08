# Emotion-detection-App

A web-based application for emotion classification using a fine-tuned DistilBERT model. The app allows users to input text and returns the predicted emotion with a confidence score and emoji.

## 📊 Overview

This application was built and deployed using:
- **DistilBERT** fine-tuned for emotion classification
- **Streamlit** for the interactive web interface
- **Docker** container for deployment
- **Hugging Face Spaces** for hosting

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

## 🚀 Deployment

The application is deployed and publicly accessible on [Hugging Face Spaces](https://huggingface.co/spaces/Ikraaaam/Emotion_detection_app).

Deployment was done using the **Docker SDK** available in Hugging Face Spaces. The Streamlit app runs inside a Docker container, and the trained BERT model is loaded from a local directory included in the container.

If you wish to explore or run the project **locally**, you can find the full source code and setup instructions in the [GitHub repository](https://github.com/Ikraaaam/emotion-detection-app).



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
| **DistilBERT**     | *0.94*  | Best overall performance         |
| XGBoost            | *0.74*  | Limited on contextual features   |
| Logistic Regression | *0.87*  | Fast but less expressive         |
| RNN (GRU)         | *0.35*  | Decent but underperformed BERT   |


📊 Why Did GRU Fail Here?

Problem	Explanation

| Problem                      | Explanation                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| 💬 Very short texts          | The RNN doesn't have enough context to capture meaningful patterns                      |
| 🧪 Not enough data           | RNNs need a large amount of examples to learn from sequential information               |
| 🧱 No pre-trained embeddings | The GRU learned everything from scratch, so it had no prior knowledge of word meanings |
| 📐 Overparameterization      | Too many neurons → leads to poor learning or convergence to the dominant class          |

## 🧪 Local Setup Instructions

To run the app locally:

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/Ikraaaam/emotion-detection-app
   cd emotion-detection-app
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the label encoder file (already included in the repo under `src/label_encoder.pkl`).

5. You **do not** need to download or retrain the BERT model. It is publicly available on Hugging Face at:
   [`Ikraaaam/bert_emotion_app`](https://huggingface.co/Ikraaaam/bert-emotion-app)

6. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📷 Screenshots

Screenshots of the deployed app (on Hugging Face or local environment).
### The App :
![image](https://github.com/user-attachments/assets/930d9037-f4b6-4ab3-a3a6-4c78a413ccd7)

### Prediction of the sentence "I'm upset" :
![image](https://github.com/user-attachments/assets/7bb8c392-ea15-4d50-9077-1b600bf9c0ef)

### The history of previous predictions :
![image](https://github.com/user-attachments/assets/cc73db24-8098-44cc-bffd-f92bbcec2e9f)

### Prediction of sentences in a txt file :
![image](https://github.com/user-attachments/assets/2b028dea-f1b5-4b4d-9cff-637fcb1e382a)


---

## 📌 Note
If you face memory issues in Hugging Face Spaces, consider reducing batch size, optimizing model size, or switching to Gradio + hosted model.

