import streamlit as st
import joblib
import pandas as pd
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch







# ✨ Emoji mapping for emotions
EMOJI_MAP = {
    "joy": "😄",
    "sadness": "😞",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# 🔹 Load model, tokenizer, encoder
#appeler les modèles depuis hugging face au lieu de le faire depuis le répertoire du code 
model = DistilBertForSequenceClassification.from_pretrained("Ikraaaam/bert-emotion-app")
tokenizer = DistilBertTokenizerFast.from_pretrained("Ikraaaam/bert-emotion-app")



#model = DistilBertForSequenceClassification.from_pretrained("bert_emotion_model")
#tokenizer = DistilBertTokenizerFast.from_pretrained("bert_emotion_model")
encoder = joblib.load("label_encoder.pkl")

# 🔹 Prediction function
def predict_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    pred_idx = np.argmax(probs)
    label = encoder.inverse_transform([pred_idx])[0]
    return label, probs

# 🌐 Streamlit interface
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🚀 BERT Emotion Detection")

# Language & instructions
st.selectbox("Choose language (currently English only)", ["English"])
st.markdown("Enter a sentence to detect its emotion:")

# Input area
text_input = st.text_input("Text to analyze")

# Predict single input
if text_input:
    label, probs = predict_emotion(text_input)
    emoji = EMOJI_MAP.get(label, "")
    st.success(f"**Predicted emotion:** {label} {emoji}")

    st.subheader("Probability distribution:")
    prob_dict = {encoder.inverse_transform([i])[0]: probs[i] for i in range(len(probs))}
    df_probs = pd.DataFrame(prob_dict.items(), columns=["Emotion", "Probability"])
    df_probs["Emoji"] = df_probs["Emotion"].map(EMOJI_MAP)
    st.bar_chart(df_probs.set_index("Emotion")["Probability"])

# 📂 Upload file for batch prediction
st.subheader("Or upload a .txt or .csv file")
file = st.file_uploader("Upload a text or CSV file", type=["txt", "csv"])

if file:
    if file.name.endswith(".txt"):
        lines = file.read().decode("utf-8").splitlines()
        df = pd.DataFrame(lines, columns=["text"])
    else:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        
    predictions = []
    for text in df["text"]:
        label, probs = predict_emotion(text)
        emoji = EMOJI_MAP.get(label, "")
        predictions.append({"text": text, "emotion": label, "emoji": emoji})

    st.subheader("Batch Predictions:")
    st.dataframe(pd.DataFrame(predictions))

# ⏲️ Prediction history (last 5)
if "history" not in st.session_state:
    st.session_state.history = []

if text_input:
    st.session_state.history.append((text_input, label))
    st.session_state.history = st.session_state.history[-5:]

if st.session_state.history:
    st.subheader("Last 5 predictions:")
    for i, (text, pred) in enumerate(reversed(st.session_state.history), 1):
        st.write(f"{i}. **{text}** → _{pred}_ {EMOJI_MAP.get(pred, '')}")
