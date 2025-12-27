
import argparse
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_melspec(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def save_spectrogram_png(S_db, out_png_path, sr=SR, hop_length=HOP_LENGTH):
    """
    Save a spectrogram array S_db to PNG path.
    Renamed to avoid shadowing boolean variables named save_png.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

def process_segment(file_path: Path, out_npy_dir: Path, out_png_dir: Path, seg_name: str,
                    offset=None, duration=None, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                    save_png=True, overwrite=False):
    """
    Loads either whole file (if duration is None) or a segment and saves .npy and optional .png.
    Returns dict with metadata (or None if segment could not be loaded).
    """
    try:
        if duration is None:
            y, _ = librosa.load(str(file_path), sr=sr, mono=True)
        else:

            y, _ = librosa.load(str(file_path), sr=sr, mono=True, offset=float(offset), duration=float(duration))
    except Exception as e:
        print(f"Could not load {file_path} (offset={offset} dur={duration}): {e}")
        return None

    if y.size == 0:
        return None


    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    S_db = make_melspec(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    npy_name = seg_name + ".npy"
    png_name = seg_name + ".png"

    npy_path = out_npy_dir / npy_name
    png_path = out_png_dir / png_name

    ensure_dir(npy_path.parent)
    ensure_dir(png_path.parent)

    if overwrite or not npy_path.exists():
        np.save(str(npy_path), S_db)
    # CALL THE RENAMED FUNCTION HERE (only if save_png is True)
    if save_png and (overwrite or not png_path.exists()):
        save_spectrogram_png(S_db, str(png_path), sr=sr, hop_length=hop_length)

    return {
        "original_filename": file_path.name,
        "segment_filename": seg_name,
        "npy_path": str(npy_path),
        "png_path": str(png_path) if save_png else "",
    }

def scan_and_process(dataset_root: Path, out_dir: Path,
                     segment_duration=None, include_last=False, min_last_duration=0.5,
                     sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                     save_png=True, overwrite=False,
                     extensions=None):
    extensions = extensions or {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    dataset_root = dataset_root.resolve()
    out_dir = out_dir.resolve()
    out_npy = out_dir / "npy"
    out_png = out_dir / "png"
    ensure_dir(out_npy)
    ensure_dir(out_png)

    rows = []
    labels = []

    # Expect immediate children of dataset_root to be instrument folders
    for label_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        labels.append(label)
        audio_files = sorted([p for p in label_dir.rglob("*") if p.suffix.lower() in extensions])
        if len(audio_files) == 0:
            print(f"Warning: no audio files found in {label_dir}")
            continue

        for file_path in tqdm(audio_files, desc=f"Processing '{label}'"):
            try:
                if segment_duration is None:
                    seg_name = f"{label}/{file_path.stem}"
                    # save under label subfolder in out dirs to keep things tidy
                    out_npy_label = out_npy / label
                    out_png_label = out_png / label
                    ensure_dir(out_npy_label)
                    ensure_dir(out_png_label)
                    meta = process_segment(file_path, out_npy_label, out_png_label, seg_name,
                                           offset=None, duration=None, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                           hop_length=hop_length, save_png=save_png, overwrite=overwrite)
                    if meta:
                        meta["label"] = label
                        rows.append(meta)
                else:
                    # compute duration (efficient)
                    total_dur = librosa.get_duration(filename=str(file_path))
                    num_full = int(total_dur // segment_duration)
                    seg_idx = 0
                    for i in range(num_full):
                        offset = i * segment_duration
                        seg_name = f"{label}/{file_path.stem}_seg{seg_idx:04d}"
                        out_npy_label = out_npy / label
                        out_png_label = out_png / label
                        ensure_dir(out_npy_label)
                        ensure_dir(out_png_label)
                        meta = process_segment(file_path, out_npy_label, out_png_label, seg_name,
                                               offset=offset, duration=segment_duration, sr=sr, n_mels=n_mels,
                                               n_fft=n_fft, hop_length=hop_length, save_png=save_png, overwrite=overwrite)
                        if meta:
                            meta["label"] = label
                            rows.append(meta)
                        seg_idx += 1
                    # leftover tail
                    tail = total_dur - (num_full * segment_duration)
                    if include_last and tail >= min_last_duration:
                        offset = num_full * segment_duration
                        seg_name = f"{label}/{file_path.stem}_seg{seg_idx:04d}"
                        out_npy_label = out_npy / label
                        out_png_label = out_png / label
                        ensure_dir(out_npy_label)
                        ensure_dir(out_png_label)
                        meta = process_segment(file_path, out_npy_label, out_png_label, seg_name,
                                               offset=offset, duration=tail, sr=sr, n_mels=n_mels,
                                               n_fft=n_fft, hop_length=hop_length, save_png=save_png, overwrite=overwrite)
                        if meta:
                            meta["label"] = label
                            rows.append(meta)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Save metadata and labels
    meta_df = pd.DataFrame(rows, columns=["original_filename", "segment_filename", "npy_path", "png_path", "label"])
    ensure_dir(out_dir)
    meta_csv = out_dir / "metadata.csv"
    meta_df.to_csv(str(meta_csv), index=False)
    labels_csv = out_dir / "labels.csv"
    pd.DataFrame({"label": sorted(set(labels))}).to_csv(str(labels_csv), index=False)

    print(f"Done. Processed {len(meta_df)} items.")
    print(f"Metadata: {meta_csv}")
    print(f"Labels: {labels_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess instrument-folder dataset to mel-spectrograms.")
    parser.add_argument("--dataset_root", required=True, help="Root folder of the dataset (contains one subfolder per instrument).")
    parser.add_argument("--out_dir", required=True, help="Output folder (contains npy/, png/, metadata.csv, labels.csv).")
    parser.add_argument("--segment_duration", type=float, default=None, help="If set, segment length in seconds (e.g., 4.0).")
    parser.add_argument("--include_last", action="store_true", help="Include final short segment if shorter than segment_duration.")
    parser.add_argument("--min_last_duration", type=float, default=0.1, help="Minimum seconds of final tail to include.")
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--n_fft", type=int, default=N_FFT)
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH)
    parser.add_argument("--no_png", action="store_true", help="If set, do not save PNG images (only npy arrays).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    scan_and_process(dataset_root=Path(args.dataset_root),
                     out_dir=Path(args.out_dir),
                     segment_duration=args.segment_duration,
                     include_last=args.include_last,
                     min_last_duration=args.min_last_duration,
                     sr=args.sr,
                     n_mels=args.n_mels,
                     n_fft=args.n_fft,
                     hop_length=args.hop_length,
                     save_png=(not args.no_png),
                     overwrite=args.overwrite)
