import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "./music_dataset"
OUTPUT_DIR = "./mel_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mel_spec(file_path, n_mels=128):
    audio, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, sr

def save_mel_image(mel, sr, path):
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

for label in os.listdir(INPUT_DIR):
    label_folder = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(label_folder):
        continue

    output_label_folder = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_folder, exist_ok=True)

    for file in os.listdir(label_folder):
        if file.endswith(".wav"):

            file_path = os.path.join(label_folder, file)

            mel, sr = extract_mel_spec(file_path)  

            save_path = os.path.join(output_label_folder, file.replace(".wav", ".png"))

            save_mel_image(mel, sr, save_path)    

            print(f"Saved {save_path}")
