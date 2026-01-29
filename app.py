import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
from fpdf import FPDF
import tempfile
import tensorflow as tf
import pandas as pd

# ===============================
# HELPER FUNCTION TO SAVE GRAPHS
# ===============================
def save_graph(fig, filename):
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ===============================
# 1. LOAD MODEL
# ===============================
model = tf.keras.models.load_model("final_cnn_model_m3.h5")

LABELS = [
    "Accordion","Acoustic_Guitar","Banjo","Bass_Guitar","Clarinet","Cymbals",
    "Dobro","Drum_set","Electro_Guitar","Floor_Tom","Harmonica","Harmonium",
    "Hi_Hats","Horn","Keyboard","Mandolin","Organ","Piano","Saxophone",
    "Shakers","Tambourine","Trombone","Trumpet","Ukulele","Violin",
    "cowbell","flute","vibraphone"
]

# ===============================
# 2. PREPROCESSING
# ===============================
SAMPLE_RATE = 22050
N_MELS = 128
FIXED_LENGTH = 128

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < FIXED_LENGTH:
        mel_db = np.pad(mel_db, ((0,0),(0,FIXED_LENGTH - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :FIXED_LENGTH]
    return mel_db.reshape(1, N_MELS, FIXED_LENGTH, 1)

# ===============================
# 3. INTENSITY VS TIME (RMS ENERGY)
# ===============================
def plot_intensity_vs_time(y, sr):
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, rms, color="#1f77b4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Intensity (RMS Energy)")
    ax.set_title("Intensity vs Time")
    ax.grid(alpha=0.3)
    return fig

# ===============================
# 4. STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🎵 InstruNet AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- DARK THEME + CSS --------
st.markdown("""
<style>
body { background-color: #0e1117; color: #ffffff; }
.stButton>button { background-color: #1f77b4; color: white; font-weight:bold; }
.stProgress>div>div>div { background-color: #1f77b4; }
[data-testid="stMetricValue"] { font-size: 22px; color:#1f77b4; }
.stDownloadButton>button { background-color:#ff7f0e; color:white; font-weight:bold; }
.final-box { background-color: #2ca02c; color:white; padding:10px; text-align:center; border-radius:10px; font-size:24px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.title("🎵 InstruNet AI – Smart Instrument Recognition")
st.markdown("CNN + Mel Spectrogram based classifier with interactive visualization")

# ===== SESSION HISTORY STORE =====
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# MULTI AUDIO UPLOAD
# ===============================
uploaded_files = st.file_uploader(
    "Upload Audio Files",
    type=["wav","mp3"],
    accept_multiple_files=True
)
st.markdown("## 🎙️ Live Audio Recording")
live_audio = st.audio_input("Record using Microphone")

# ===============================
# PROCESS LIVE AUDIO
# ===============================
if live_audio is not None:
    st.divider()
    st.subheader("🎙️ Live Recorded Audio")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(live_audio.read())
        live_path = tmp.name

    st.audio(live_path)
    y, sr = librosa.load(live_path, sr=None)

    # -------- GRAPHS IN TABS --------
    fig, fig2, fig3 = None, None, None
    tabs = st.tabs(["Waveform", "Spectrogram", "Intensity vs Time"])
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform (Live)")
        st.pyplot(fig)
    with tabs[1]:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        ax2.set_title("Spectrogram (Live)")
        st.pyplot(fig2)
    with tabs[2]:
        fig3 = plot_intensity_vs_time(y, sr)
        st.pyplot(fig3)

    # -------- PREDICTION --------
    features = extract_features(live_path)
    preds = model.predict(features)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    result_dict = {LABELS[i]: round(float(preds[i])*100,2) for i in top3}

    st.markdown(
        f"<div class='final-box'>🎯 Predicted Instrument (Live): {list(result_dict.keys())[0]}</div>",
        unsafe_allow_html=True
    )

    st.subheader("🤖 Top-3 Predictions (Live)")
    cols = st.columns(3)
    colors = ["#1f77b4","#ff7f0e","#2ca02c"]
    for idx, (inst, conf) in enumerate(result_dict.items()):
        with cols[idx]:
            st.markdown(f"<div style='text-align:center; font-size:20px; color:{colors[idx]}'>{inst}</div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{conf}%")
            st.progress(int(conf))

    # Save graphs
    waveform_img = "live_waveform.png"
    spectrogram_img = "live_spectrogram.png"
    intensity_img = "live_intensity.png"
    save_graph(fig, waveform_img)
    save_graph(fig2, spectrogram_img)
    save_graph(fig3, intensity_img)

    # -------- LIVE JSON DOWNLOAD --------
    live_report = {"audio_file":"Live_Recording.wav", "predictions": result_dict}
    st.download_button(
        label="⬇️ Download Live JSON Report",
        data=json.dumps(live_report, indent=4),
        file_name="live_audio_report.json",
        mime="application/json"
    )

    # -------- ADD TO HISTORY --------
    st.session_state.history.append({
        "File":"Live_Recording",
        "Prediction":list(result_dict.keys())[0],
        "Confidence":list(result_dict.values())[0]
    })

    # -------- LIVE PDF EXPORT --------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial","B",18)
    pdf.cell(0,12,"InstruNet AI - Live Audio Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Audio Source: Live Microphone Recording", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(44,160,44)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Arial","B",14)
    pdf.cell(0,12,f" Predicted Instrument: {list(result_dict.keys())[0]} ({list(result_dict.values())[0]}%)", ln=True, fill=True)
    pdf.ln(8)
    pdf.set_text_color(0,0,0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Top-3 Predictions:", ln=True)
    pdf.ln(3)
    for inst, conf in result_dict.items():
        pdf.cell(0,8,f"- {inst} : {conf}%", ln=True)
    pdf.ln(8)
    pdf.cell(0,10,"Waveform", ln=True); pdf.ln(3); pdf.image(waveform_img, x=15,w=180); pdf.ln(10)
    pdf.cell(0,10,"Spectrogram", ln=True); pdf.ln(3); pdf.image(spectrogram_img, x=15,w=180); pdf.ln(10)
    pdf.cell(0,10,"Intensity vs Time", ln=True); pdf.ln(3); pdf.image(intensity_img, x=15,w=180); pdf.ln(10)
    live_pdf_path = "live_audio_full_report.pdf"
    pdf.output(live_pdf_path)
    with open(live_pdf_path,"rb") as f:
        st.download_button("⬇️ Download Full Live PDF Report", f, file_name=live_pdf_path, mime="application/pdf")

# ===============================
# PROCESS UPLOADED FILES
# ===============================
for uploaded_file in uploaded_files:
    st.divider()
    st.subheader(f"🎧 {uploaded_file.name}")
    ext = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
    st.audio(audio_path, format=f"audio/{ext}")
    y, sr = librosa.load(audio_path, sr=None)

    # -------- GRAPHS IN TABS --------
    tabs = st.tabs(["Waveform", "Spectrogram", "Intensity vs Time"])
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10,3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)
        save_graph(fig, f"{uploaded_file.name}_waveform.png")
    with tabs[1]:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(10,3))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        ax2.set_title("Spectrogram")
        st.pyplot(fig2)
        save_graph(fig2, f"{uploaded_file.name}_spectrogram.png")
    with tabs[2]:
        fig3 = plot_intensity_vs_time(y, sr)
        st.pyplot(fig3)
        save_graph(fig3, f"{uploaded_file.name}_intensity.png")

    # -------- PREDICTION --------
    features = extract_features(audio_path)
    preds = model.predict(features)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    result_dict = {LABELS[i]: round(float(preds[i])*100,2) for i in top3}
    st.markdown(f"<div class='final-box'>🎯 Predicted Instrument: {list(result_dict.keys())[0]}</div>", unsafe_allow_html=True)
    st.subheader("🤖 Top-3 Predictions")
    cols = st.columns(3)
    colors = ["#1f77b4","#ff7f0e","#2ca02c"]
    for idx,(inst,conf) in enumerate(result_dict.items()):
        with cols[idx]:
            st.markdown(f"<div style='text-align:center;font-size:20px;color:{colors[idx]}'>{inst}</div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{conf}%")
            st.progress(int(conf))

    # ===== HISTORY =====
    st.session_state.history.append({"File":uploaded_file.name,"Prediction":list(result_dict.keys())[0],"Confidence":list(result_dict.values())[0]})

    # ===== JSON REPORT =====
    report = {"audio_file":uploaded_file.name,"predictions":result_dict}
    st.download_button(f"Download JSON Report", data=json.dumps(report, indent=4), file_name=f"{uploaded_file.name}_report.json", mime="application/json")

    # ===== PDF EXPORT =====
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial","B",18)
    pdf.cell(0,12,"InstruNet AI - Audio Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,f"Audio File: {uploaded_file.name}", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(44,160,44)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Arial","B",14)
    pdf.cell(0,12,f" Predicted Instrument: {list(result_dict.keys())[0]} ({list(result_dict.values())[0]}%)", ln=True, fill=True)
    pdf.ln(8)
    pdf.set_text_color(0,0,0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Top-3 Predictions:", ln=True)
    pdf.ln(3)
    for inst, conf in result_dict.items():
        pdf.cell(0,8,f"- {inst} : {conf}%", ln=True)
    pdf.ln(8)
    pdf.cell(0,10,"Waveform", ln=True); pdf.ln(3); pdf.image(f"{uploaded_file.name}_waveform.png", x=15,w=180); pdf.ln(10)
    pdf.cell(0,10,"Spectrogram", ln=True); pdf.ln(3); pdf.image(f"{uploaded_file.name}_spectrogram.png", x=15,w=180); pdf.ln(10)
    pdf.cell(0,10,"Intensity vs Time", ln=True); pdf.ln(3); pdf.image(f"{uploaded_file.name}_intensity.png", x=15,w=180); pdf.ln(10)
    upload_pdf_path = f"{uploaded_file.name}_full_report.pdf"
    pdf.output(upload_pdf_path)
    with open(upload_pdf_path,"rb") as f:
        st.download_button("⬇️ Download Full PDF Report", f, file_name=upload_pdf_path, mime="application/pdf")

# ===============================
# GLOBAL HISTORY TABLE
# ===============================
st.divider()
st.subheader("📜 Prediction History")
if len(st.session_state.history)>0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Full History CSV", data=df.to_csv(index=False), file_name="prediction_history.csv", mime="text/csv")
else:
    st.info("No predictions yet")
