import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tempfile
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "instrument_cnn_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE = 22050

st.set_page_config(
    page_title="Music Instrument Recognition",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_melspectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

    S_db = tf.image.resize(S_db[..., np.newaxis], (128, 256))
    S_db = tf.expand_dims(S_db, axis=0)

    return S_db, y, sr

# ---------------- UI ----------------
st.title("🎶 Music Instrument Recognition System")
st.markdown("Upload an audio file and detect the musical instrument using CNN")

uploaded_file = st.file_uploader(
    "Upload Audio File (.wav / .mp3)",
    type=["wav", "mp3"]
)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path)

    spec, y, sr = audio_to_melspectrogram(audio_path)

    # Prediction
    predictions = model.predict(spec)[0]
    pred_index = np.argmax(predictions)
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = predictions[pred_index]

    # Display result
    st.success(f"🎵 **Predicted Instrument:** {pred_label}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}")

    # Confidence bar chart
    st.subheader("Prediction Confidence")
    fig, ax = plt.subplots()
    ax.barh(label_encoder.classes_, predictions)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Spectrogram display

    # ---------------- Spectrogram Display ----------------
    st.subheader("Mel-Spectrogram")

    import librosa.display
    import matplotlib.pyplot as plt

    # Create figure and axis
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot spectrogram
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=ax2
    )

    # Add colorbar correctly
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")

    ax2.set_title("Mel-Spectrogram")

    st.pyplot(fig2)


    os.remove(audio_path)
