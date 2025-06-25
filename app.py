
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition App")
st.markdown("Upload a `.wav` audio file and get the predicted emotion!")

# Load model and label encoder
@st.cache_resource
def load_artifacts():
    model = load_model("emotion_model_filtered_final.h5")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder
# changes made
model = load_model("emotion_model_filtered_final.h5", compile=False)

#File upload
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    try:
        # Audio preprocessing
        audio, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]

        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # shape: (1, 40, 130, 1)

        # Prediction
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_index])[0]

        st.success(f"🧠 Predicted Emotion: **{emotion.capitalize()}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
