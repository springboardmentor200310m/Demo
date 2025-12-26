import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import load_audio, generate_mel_spectrogram
from model import build_model

# Load model (cached)
@st.cache_resource
def load_trained_model():
    model_path = 'models/instrunet_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # Return a dummy model if not trained yet
        return build_model((128, 128, 1), 5)

def main():
    st.title("InstruNet AI - Instrument Recognition")
    st.write("Upload an audio file to detect instruments.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Save to temp file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process
        st.write("Processing...")
        y, sr = load_audio("temp_audio.wav")
        
        if y is not None:
            # Display Waveform
            st.subheader("Waveform")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)
            
            # Generate Spectrogram
            S_dB = generate_mel_spectrogram(y, sr)
            
            # Display Spectrogram
            st.subheader("Mel Spectrogram")
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            
            # Predict
            st.write("Detecting instruments...")
            model = load_trained_model()
            
            # Resize/Crop spectrogram to match model input (dummy logic for now)
            # In real app, we'd slice the spectrogram into windows
            input_shape = model.input_shape[1:] # (128, 128, 1)
            
            # Dummy resizing for demo
            if S_dB.shape[1] >= input_shape[1]:
                S_input = S_dB[:, :input_shape[1]]
            else:
                S_input = np.pad(S_dB, ((0, 0), (0, input_shape[1] - S_dB.shape[1])))
                
            S_input = np.expand_dims(S_input, axis=-1) # Add channel dim
            S_input = np.expand_dims(S_input, axis=0) # Add batch dim
            
            preds = model.predict(S_input)
            
            # Display Results
            st.subheader("Detected Instruments")
            classes = ['Piano', 'Guitar', 'Drums', 'Violin', 'Flute']
            
            for i, label in enumerate(classes):
                st.write(f"{label}: {preds[0][i]*100:.2f}%")
                st.progress(float(preds[0][i]))
                
        else:
            st.error("Error loading audio file.")

if __name__ == '__main__':
    main()
