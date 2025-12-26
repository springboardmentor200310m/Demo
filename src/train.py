import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from model import InstrumentCNN

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(BASE_DIR, "..", "spectrograms")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "instrument_cnn.pth")

CLASSES = sorted(os.listdir(SPEC_DIR))
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.transform(img)
        return img, self.labels[idx]

# ---------------- LOAD DATA ----------------
image_paths = []
labels = []

for idx, inst in enumerate(CLASSES):
    inst_dir = os.path.join(SPEC_DIR, inst)
    for img in os.listdir(inst_dir):
        image_paths.append(os.path.join(inst_dir, img))
        labels.append(idx)

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = SpectrogramDataset(train_imgs, train_labels, train_transform)
val_dataset = SpectrogramDataset(val_imgs, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = InstrumentCNN(NUM_CLASSES).to(DEVICE)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    val_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Avg Train Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model saved:", MODEL_PATH)
