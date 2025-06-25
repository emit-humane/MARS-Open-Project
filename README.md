# MARS-Open-Project
Design and implement a robust pipeline to  perform emotion classification on speech data 

# 🎤 Emotion Classification from Speech using Deep Learning

This project implements a full pipeline to classify emotions from speech audio using the **RAVDESS dataset**. We extract audio features using `librosa`, and train a deep neural network (CNN + LSTM) to classify emotions like **happy**, **sad**, **angry**, and more.

---

## 📁 Dataset

We use the **RAVDESS dataset**: [Download on Zenodo](https://zenodo.org/record/1188976)

Files used:
- `Audio_Speech_Actors_01-24`
- `Audio_Song_Actors_01-24`

Each `.wav` file is labeled based on filename metadata, which encodes the speaker’s emotion.

---

## 🎯 Project Objective

Build an end-to-end machine learning pipeline that:
- Preprocesses `.wav` files using audio features (MFCC, Chroma, Mel)
- Trains a CNN + LSTM model for emotion recognition
- Provides a user-friendly web interface using Streamlit
- Supports CLI-based prediction using a Python script

---

## 🧠 Model Overview

### 📊 Feature Extraction
We extract the following features using `librosa`:
- 40 MFCCs (Mel-Frequency Cepstral Coefficients)
- Chroma features
- Mel Spectrogram

### 🧱 Model Architecture
Your model combines CNN and LSTM layers:
- 3 × 1D Convolutional layers (Conv1D + MaxPooling + BatchNorm)
- 3 × LSTM layers
- Fully connected dense layers
- Final output layer with softmax activation

```python
Conv1D → MaxPool → BatchNorm → Dropout
→ (repeat)
→ LSTM (×3) → Dense(128 → 64 → 32 → softmax)
📈 Performance
Validation Accuracy: ~85%

F1-Score: > 80%

Evaluated using a confusion matrix and classification_report
![image](https://github.com/user-attachments/assets/f0ddf64b-b3ac-418c-a177-137296be4f47)
![image](https://github.com/user-attachments/assets/f0ddf64b-b3ac-418c-a177-137296be4f47)


🚀 Running the Project
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Predict Emotion from a .wav file
python test_model.py path_to_audio.wav
3️⃣ Run Web App (Streamlit)
streamlit run app.py
Upload a .wav file

Get real-time emotion prediction

🔧 Project Structure
emotion-classification/
├── model_training.ipynb            # Colab notebook with full training pipeline
├── emotion_classification_model.h5 # Trained model
├── app.py                          # Streamlit web app
├── test_model.py                   # CLI prediction script
├── requirements.txt                # Dependencies
├── README.md                       # This file
📹 [Demo Video](https://drive.google.com/file/d/1adyPLr-APf5GupYBAbAqJzR5N_2CpJsU/view?usp=drive_link)
🎬 Watch the demo — shows the model training, Streamlit app, and command-line testing.

✍️ Author
Rudra Sharma

MARS Open Projects 2025 — Project 1 Submission

Contact: rudra310sharma@gmail.com

