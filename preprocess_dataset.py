# preprocess_dataset.py
# Prepares IRMAS audio dataset into mel-spectrogram PNGs + numpy arrays + labels.csv
# Works in VS Code or Kaggle

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# -----------------------
# CONFIGURATION
# -----------------------

RAW_DATA_DIR = "dataset"              # Input folder containing piano/, guitar/, violin/
OUTPUT_IMG_DIR = "spectrograms"       # Output folder for image spectrograms
OUTPUT_ARR_DIR = "arrays"             # Output folder for .npy spectrogram arrays
LABELS_CSV = "labels.csv"             # Labels CSV file

SAMPLE_RATE = 22050                   # Audio target sample rate
DURATION = 3.0                        # Target duration (seconds)
FIX_LENGTH = int(SAMPLE_RATE * DURATION)  # Number of samples in fixed-length audio

N_MELS = 128                          # Mel bands
FMAX = 8000                           # Frequency upper bound
FIGSIZE = (2.56, 2.56)                # Spectrogram image size
CMAP = "inferno"                      # Color map for spectrograms

# Optional: Downsample large classes to this number
BALANCE_TO = None   # e.g., 500 or None to keep all


# -----------------------
# HELPER FUNCTIONS
# -----------------------

def list_classes(raw_dir):
    """Return sorted list of class names (subdirectories)."""
    return [d for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))]


def load_audio(path, sample_rate=SAMPLE_RATE, fix_length=FIX_LENGTH):
    """Load audio as mono, trim silence, normalize, and pad/trim to fixed length."""
    y, sr = librosa.load(path, sr=sample_rate, mono=True)

    # normalize
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max()

    # trim silence
    y, _ = librosa.effects.trim(y)

    # pad or trim to fixed length
    if len(y) < fix_length:
        y = np.pad(y, (0, fix_length - len(y)), mode="constant")
    else:
        y = y[:fix_length]

    return y, sr


def make_mel(y, sr, n_mels=N_MELS, fmax=FMAX):
    """Convert audio waveform to mel-spectrogram."""
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def save_spec_image(spec, out_path):
    """Save mel-spectrogram as PNG image."""
    plt.figure(figsize=FIGSIZE)
    librosa.display.specshow(spec, sr=SAMPLE_RATE, cmap=CMAP, fmax=FMAX)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_array(arr, out_path):
    """Save numpy array."""
    np.save(out_path, arr)


def balance_files(file_list, keep_n):
    """Randomly keep only limited number of samples per class."""
    import random
    if keep_n is None or len(file_list) <= keep_n:
        return file_list
    return random.sample(file_list, keep_n)


# -----------------------
# MAIN PREPROCESSING PIPELINE
# -----------------------

def preprocess_all():
    classes = list_classes(RAW_DATA_DIR)
    print("Classes found:", classes)

    all_labels = []

    # Collect files from each class
    class_files = {}
    for cls in classes:
        cls_path = os.path.join(RAW_DATA_DIR, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith((".wav", ".mp3", ".aiff", ".ogg", ".flac"))]
        class_files[cls] = files

    # Optional downsampling
    if BALANCE_TO is not None:
        for cls in classes:
            class_files[cls] = balance_files(class_files[cls], BALANCE_TO)

    # Create output dirs
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_IMG_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ARR_DIR, cls), exist_ok=True)

    # Process each class
    for cls in classes:
        print(f"\nProcessing {cls} ({len(class_files[cls])} files)...")

        for fname in tqdm(class_files[cls]):
            raw_path = os.path.join(RAW_DATA_DIR, cls, fname)

            try:
                y, sr = load_audio(raw_path)
            except Exception as e:
                print("Skipping:", raw_path, "Error:", e)
                continue

            # mel spectrogram
            spec_db = make_mel(y, sr)

            # output filenames
            base = os.path.splitext(fname)[0]
            img_out = os.path.join(OUTPUT_IMG_DIR, cls, f"{cls}__{base}.png")
            arr_out = os.path.join(OUTPUT_ARR_DIR, cls, f"{cls}__{base}.npy")

            # save spectrogram image + array
            save_spec_image(spec_db, img_out)
            save_array(spec_db, arr_out)

            # add label entry
            all_labels.append([f"{cls}/{cls}__{base}.png", cls])

    # Save labels CSV
    df = pd.DataFrame(all_labels, columns=["filename", "label"])
    df.to_csv(LABELS_CSV, index=False)

    print("\nPreprocessing complete!")
    print("Images saved to:", OUTPUT_IMG_DIR)
    print("Arrays saved to:", OUTPUT_ARR_DIR)
    print("Labels saved to:", LABELS_CSV)


if __name__ == "__main__":
    preprocess_all()
