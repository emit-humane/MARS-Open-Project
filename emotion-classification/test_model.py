# test_model.py

import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys

# Load model
model = load_model('emotion_classification_model.h5')

# Define your emotion labels (same order as training)
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
le = LabelEncoder()
le.fit(emotion_labels)

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

# Predict emotion
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    emotion = le.inverse_transform([predicted_class])[0]
    return emotion

# Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <audio_file_path>")
    else:
        file_path = sys.argv[1]
        print("Predicted Emotion:", predict_emotion(file_path))
