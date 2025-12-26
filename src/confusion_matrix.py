from sklearn.metrics import accuracy_score, classification_report
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import InstrumentCNN

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "spectrograms")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "instrument_cnn.pth")

device = torch.device("cpu")

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------------
# DATASET
# -----------------------------
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes

# -----------------------------
# LOAD MODEL
# -----------------------------
model = InstrumentCNN(len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Instrument Classification Confusion Matrix")

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)

print("\n📊 Model Evaluation Metrics")
print(f"✅ Accuracy: {accuracy:.4f}")

print("\n📄 Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)
plt.show()

