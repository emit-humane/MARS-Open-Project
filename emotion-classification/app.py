import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler

# --- Load model and scaler ---
model = tf.keras.models.load_model("emotion_classification_model.h5")
scaler = joblib.load("scaler.pkl")  # Save using joblib.dump(scaler, "scaler.pkl")

# --- Emotion labels (same order as one-hot) ---
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# --- Feature extraction ---
def extract_features(data, sample_rate):
    result = np.array([])
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_input_features(file):
    data, sample_rate = librosa.load(file, duration=2.5, offset=0.6)
    raw_features = extract_features(data, sample_rate)
    scaled = scaler.transform([raw_features])  # shape (1, 162)
    return np.expand_dims(scaled, axis=2)      # shape (1, 162, 1)

# --- Streamlit UI ---
st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        features = get_input_features(uploaded_file)
        prediction = model.predict(features)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        st.markdown(f"### Predicted Emotion: **{predicted_emotion}**")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
