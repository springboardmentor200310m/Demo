import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DATA_ROOT = "IRMAS-TrainingData"
OUT_ROOT = "outputs/spectrogram_images"
os.makedirs(OUT_ROOT, exist_ok=True)

labels = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]


for label in labels:
    src_folder = os.path.join(DATA_ROOT, label)
    dst_folder = os.path.join(OUT_ROOT, label)
    os.makedirs(dst_folder, exist_ok=True)

    for file in tqdm(os.listdir(src_folder)[:50], desc=f"Processing {label}"):
        if file.endswith(".wav"):
            wav_path = os.path.join(src_folder, file)
            y, sr = librosa.load(wav_path, sr=22050)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=(2.5, 2.5))
            plt.axis('off')
            librosa.display.specshow(S_db, sr=sr, cmap='magma')
            out_path = os.path.join(dst_folder, file.replace(".wav", ".png"))
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()
