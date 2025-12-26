# app.py
import os
import tempfile
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import onnxruntime as ort
import streamlit as st

# ==== Constants (your values) ====
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
SPEC_LEN = 128
INSTR_FAMILY_MAP = {
    0: 'Accordion', 1: 'Acoustic_Guitar', 2: 'Banjo', 3: 'Bass_Guitar', 4: 'Clarinet',
    5: 'Cymbals', 6: 'Dobro', 7: 'Drum_set', 8: 'Electro_Guitar', 9: 'Floor_Tom',
    10: 'Harmonica', 11: 'Harmonium', 12: 'Hi_Hats', 13: 'Horn', 14: 'Keyboard',
    15: 'Mandolin', 16: 'Organ', 17: 'Piano', 18: 'Saxophone', 19: 'Shakers',
    20: 'Tambourine', 21: 'Trombone', 22: 'Trumpet', 23: 'Ukulele', 24: 'Violin',
    25: 'cowbell', 26: 'flute', 27: 'vibraphone'
}

# ==== ONNX wrapper ====
class NsynthOnnxWrapper:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=True,
            power=2.0,
        )
        self.to_db = AmplitudeToDB(stype="power")

    def preprocess_wav(self, wav_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=SAMPLE_RATE
            )
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = self.mel(waveform)
        spec = self.to_db(spec).squeeze(0)

        if spec.size(1) < SPEC_LEN:
            pad = SPEC_LEN - spec.size(1)
            spec = torch.nn.functional.pad(spec, (0, pad))
        elif spec.size(1) > SPEC_LEN:
            spec = spec[:, :SPEC_LEN]

        mean = spec.mean()
        std = spec.std() + 1e-6
        spec = (spec - mean) / std
        spec = spec.unsqueeze(0).unsqueeze(0)  # (1,1,128,128)

        return spec.numpy().astype("float32")

    def predict(self, wav_path: str):
        x = self.preprocess_wav(wav_path)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_name = INSTR_FAMILY_MAP.get(pred_idx, f"class_{pred_idx}")
        return pred_idx, pred_name, probs

# ==== Streamlit UI ====
st.set_page_config(page_title="Instrument Identifier", layout="centered")

st.title("🎵 Musical Instrument Identifier")
st.write("Upload a WAV file and the model will predict the instrument.")

# Path to your ONNX model (put the file beside app.py)
MODEL_PATH = "nsynth_instrument_family_cnn.onnx"

@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

wrapper = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Play audio
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict instrument"):
        pred_idx, pred_name, probs = wrapper.predict(tmp_path)

        st.markdown(f"### Predicted instrument: **{pred_name}**")
        st.write(f"Class index: {pred_idx}")

        # Show top‑5 probabilities
        top_k = 5
        top_indices = np.argsort(probs)[::-1][:top_k]
        st.subheader("Top predictions")
        for i in top_indices:
            st.write(f"{INSTR_FAMILY_MAP.get(int(i), i)}: {probs[i]:.3f}")

    # Clean up temp file when session ends (optional)
    # os.remove(tmp_path)
else:
    st.info("Please upload a WAV file to begin.")
