import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

from model import InstrumentCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "instrument_cnn.pth")
AUDIO_PATH = os.path.join(BASE_DIR, "..", "test_audio", "102_Ecelloarp_SP_01_376.wav")

CLASSES = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db)
    plt.axis("off")
    temp_img = "temp.png"
    plt.savefig(temp_img, bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open(temp_img).convert("L")
    os.remove(temp_img)
    return img

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = InstrumentCNN(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

img = transform(audio_to_spectrogram(AUDIO_PATH)).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(img), dim=1)[0]
    top3 = torch.topk(probs, 3)

print("\n🎵 Top-3 Predictions:")
for i in range(3):
    idx = top3.indices[i].item()
    print(f"{i+1}. {CLASSES[idx]} ({top3.values[i]:.2f})")
