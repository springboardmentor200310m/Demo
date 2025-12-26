import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

print("🔥 preprocess.py STARTED")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
SAVE_DIR = os.path.join(BASE_DIR, "..", "spectrograms")

print("BASE_DIR =", BASE_DIR)
print("DATA_DIR =", DATA_DIR)
print("SAVE_DIR =", SAVE_DIR)

def create_spectrogram(audio_path, save_path):
    print("  🎵 Processing:", audio_path)

    y, sr = librosa.load(audio_path, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(2, 2))
    librosa.display.specshow(mel_db)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("  🖼️ Saved:", save_path)

print("🔥 Reached before __main__")

if __name__ == "__main__":
    print("🚀 ENTERED __main__ BLOCK")

    if not os.path.exists(DATA_DIR):
        print("❌ DATA_DIR DOES NOT EXIST")
        exit()

    os.makedirs(SAVE_DIR, exist_ok=True)

    instruments = os.listdir(DATA_DIR)
    print("🎼 Instruments found:", instruments)

    for instrument in instruments:
        inst_path = os.path.join(DATA_DIR, instrument)
        save_inst_path = os.path.join(SAVE_DIR, instrument)

        if not os.path.isdir(inst_path):
            print("⚠️ Skipping non-folder:", instrument)
            continue

        os.makedirs(save_inst_path, exist_ok=True)
        files = os.listdir(inst_path)

        print(f"➡️ {instrument}: {len(files)} files")

        for file in files:
            if file.lower().endswith(".wav"):
                audio_file = os.path.join(inst_path, file)
                img_file = os.path.join(
                    save_inst_path,
                    file.replace(".wav", ".png")
                )
                create_spectrogram(audio_file, img_file)

    print("✅ ALL DONE")
