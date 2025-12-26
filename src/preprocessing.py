import librosa
import numpy as np
import os

def load_audio(file_path, sr=22050, duration=None):
    """
    Load an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        duration (float): Duration to load in seconds.
        
    Returns:
        np.ndarray: Audio time series.
        int: Sampling rate.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def generate_mel_spectrogram(y, sr, n_mels=128, fmax=8000):
    """
    Generate a Mel spectrogram from an audio time series.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        n_mels (int): Number of Mel bands.
        fmax (int): Maximum frequency.
        
    Returns:
        np.ndarray: Mel spectrogram (dB scale).
    """
    if y is None:
        return None
        
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def save_spectrogram(S_dB, output_path):
    """
    Save spectrogram as a numpy array.
    
    Args:
        S_dB (np.ndarray): Mel spectrogram in dB.
        output_path (str): Path to save the .npy file.
    """
    np.save(output_path, S_dB)
