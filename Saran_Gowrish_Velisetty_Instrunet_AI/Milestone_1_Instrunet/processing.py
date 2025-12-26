import os
import numpy as np
import cv2 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_ROOT_DIR = 'mel_images/'
IMAGE_EXT = '.png'
N_MELS = 128          
MAX_PAD_LEN = 128

files_found = 0
files_processed_ok = 0

def load_spectrogram_image(file_path):
    global files_processed_ok
    
    try:
        img = cv2.imread(file_path, 0) 
        
        if img is None:
            print(f"DEBUG FAIL: Failed to read image using cv2.imread: {file_path}")
            return None

        img = cv2.resize(img, (MAX_PAD_LEN, N_MELS)) 
        img = img.astype('float32') / 255.0
        files_processed_ok += 1
        return img[..., np.newaxis]
        
    except Exception as e:
        print(f"DEBUG FAIL: Unexpected error processing {file_path}: {e}")
        return None 

X_file_paths = []
y_labels_list = [] 

print(f"Scanning directory: {DATASET_ROOT_DIR}")

for instrument_name in os.listdir(DATASET_ROOT_DIR):
    instrument_path = os.path.join(DATASET_ROOT_DIR, instrument_name)
    
    if os.path.isdir(instrument_path):
        for file_name in os.listdir(instrument_path):
            if file_name.lower().endswith(IMAGE_EXT):
                file_path = os.path.join(instrument_path, file_name)
                X_file_paths.append(file_path)
                y_labels_list.append([instrument_name]) 
                files_found += 1

if files_found == 0:
    print(f"\nCRITICAL ERROR: Found 0 files matching *{IMAGE_EXT} in subfolders of {DATASET_ROOT_DIR}. Check your path or IMAGE_EXT.")
    exit()

mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y_labels_list)
NUM_CLASSES = len(mlb.classes_)


X_features = []
y_final = []

print(f"Starting feature loading ({files_found} files expected)...")
for i, file_path in enumerate(X_file_paths):
    img_data = load_spectrogram_image(file_path)
    
    if img_data is not None:
        X_features.append(img_data)
        y_final.append(y_binarized[i])


if files_processed_ok != files_found:
    print(f"\nWARNING: Only {files_processed_ok} of {files_found} files were successfully processed.")

X = np.array(X_features)
y = np.array(y_final)

if len(X) == 0:
    print("\nCRITICAL ERROR: X is empty (length 0). All files failed the validation or read step.")
    exit() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


print("\n--- Summary ---")
print(f"Total processed samples: {len(X)}")
print(f"Number of classes (Instruments): {NUM_CLASSES}")
print(f"Class names: {list(mlb.classes_)}")

np.savez_compressed(
    "instrument_melspec_dataset.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print("Saved as instrument_melspec_dataset.npz")