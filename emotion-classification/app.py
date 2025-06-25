# app.py

import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model('emotion_classification_model.h5')

# Label encoder
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
le = LabelEncoder()
le.fit(emotion_labels)

# Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

# Streamlit UI
st.title("🎵 Emotion Classifier from Speech")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file).reshape(1, -1)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    emotion = le.inverse_transform([predicted_class])[0]
    st.success(f"Predicted Emotion: **{emotion.upper()}**")
