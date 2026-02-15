"""
Procesa la carpeta de videos (videos/PALABRA/1.mp4, 2.mp4, ...) y genera la estructura
que espera el Colab de entrenamiento:

  output_dir/
  └── data/
      ├── train/
      │   ├── labels.csv
      │   └── poses/
      │       ├── 0.npy
      │       └── ...
      └── test/
          ├── labels.csv
          └── poses/
              ├── 0.npy
              └── ...

Un porcentaje de las muestras (TEST_SPLIT, ej. 20%%) va a test; el resto a train.
La división es estratificada por etiqueta (cada clase aporta ~TEST_SPLIT a test).
"""

import os
import random
import argparse
from collections import defaultdict

import pandas as pd
import mediapipe as mp

from extract_keypoints import extract_keypoints_from_video

# --- Constantes que podés editar ---
# Ruta donde están los videos (subcarpetas por palabra: AGUA, CHAU, etc.)
VIDEOS_DIR = "D:/Archivos/Documentos/Universidad/Trabajo Final/videos-20260120T020840Z-3-001/videos"
# Límite de carpetas a procesar (None = todas). Útil para probar con pocas (ej. 3).
MAX_FOLDERS = 3
# Porcentaje de muestras que van a test (0.2 = 20%%). El resto va a train.
TEST_SPLIT = 0.2
# Semilla para el split reproducible
RANDOM_STATE = 42

# Extensiones de video que buscamos
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_video_files(videos_dir, max_folders=None):
    """
    Recorre videos_dir y devuelve una lista de (label, video_path) donde label
    es el nombre de la subcarpeta (ej. AGUA) y video_path la ruta absoluta al archivo.
    Orden: por label, luego por nombre de archivo (1.mp4, 2.mp4, ...).
    Si max_folders es un entero, solo se incluyen videos de las primeras max_folders carpetas.
    """
    items = []
    folders_seen = 0
    for label in sorted(os.listdir(videos_dir)):
        folder = os.path.join(videos_dir, label)
        if not os.path.isdir(folder):
            continue
        if max_folders is not None and folders_seen >= max_folders:
            break
        folders_seen += 1
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            items.append((label, os.path.join(folder, fname)))
    return items


def stratified_train_test_split(video_list, test_ratio=0.2, random_state=42):
    """
    Divide la lista (label, video_path) en train y test de forma estratificada por etiqueta.
    Devuelve (train_list, test_list).
    """
    if test_ratio <= 0:
        return video_list, []
    if test_ratio >= 1:
        return [], video_list
    by_label = defaultdict(list)
    for item in video_list:
        label, _ = item
        by_label[label].append(item)
    rng = random.Random(random_state)
    train_list = []
    test_list = []
    for label in sorted(by_label.keys()):
        lst = by_label[label]
        rng.shuffle(lst)
        n = len(lst)
        n_test = int(round(n * test_ratio))
        n_test = max(0, min(n_test, n - 1))  # al menos 1 en train si n >= 1
        test_list.extend(lst[:n_test])
        train_list.extend(lst[n_test:])
    return train_list, test_list


def run_extraction(
    videos_dir,
    output_dir,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    skip_existing=False,
    max_folders=None,
    test_split=0.2,
    random_state=42,
):
    """
    Genera data/train/ y data/test/ (cada uno con poses/ y labels.csv) en output_dir.
    Un porcentaje test_split de las muestras va a test (división estratificada por etiqueta).

    Args:
        videos_dir: Carpeta raíz de videos (contiene subcarpetas AGUA, CHAU, ...).
        output_dir: Carpeta raíz de salida.
        min_detection_confidence: Pasado a MediaPipe.
        min_tracking_confidence: Pasado a MediaPipe.
        skip_existing: Si True, no reprocesar videos cuyo .npy ya exista.
        max_folders: Si es int, solo procesar las primeras N carpetas (None = todas).
        test_split: Proporción de muestras para test (ej. 0.2 = 20%%).
        random_state: Semilla para el split reproducible.
    """
    train_dir = os.path.join(output_dir, "data", "train")
    test_dir = os.path.join(output_dir, "data", "test")
    train_poses_dir = os.path.join(train_dir, "poses")
    test_poses_dir = os.path.join(test_dir, "poses")
    os.makedirs(train_poses_dir, exist_ok=True)
    os.makedirs(test_poses_dir, exist_ok=True)

    video_list = find_video_files(videos_dir, max_folders=max_folders)
    if not video_list:
        print(f"No se encontraron videos en {videos_dir} (subcarpetas con {VIDEO_EXTENSIONS})")
        return
    if max_folders is not None:
        print(f"Procesando solo las primeras {max_folders} carpetas ({len(video_list)} videos).")

    train_list, test_list = stratified_train_test_split(
        video_list, test_ratio=test_split, random_state=random_state
    )
    print(f"Split: {len(train_list)} train, {len(test_list)} test ({100 * test_split:.0f}% test).")

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    def process_split(split_list, poses_dir, labels_path, split_name):
        rows = []
        for sample_id, (label, video_path) in enumerate(split_list):
            npy_path = os.path.join(poses_dir, f"{sample_id}.npy")
            if skip_existing and os.path.isfile(npy_path):
                print(f"[skip] {split_name} {video_path} -> {npy_path}")
                rows.append({"ID": sample_id, "Label": label})
                continue
            print(f"[{split_name}] {sample_id + 1}/{len(split_list)} {label} - {os.path.basename(video_path)}")
            try:
                extract_keypoints_from_video(
                    video_path,
                    output_path=npy_path,
                    holistic=holistic,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                rows.append({"ID": sample_id, "Label": label})
            except Exception as e:
                print(f"  Error: {e}")
        df = pd.DataFrame(rows)
        df.to_csv(labels_path, index=False)
        print(f"  {split_name}: {labels_path} ({len(df)} muestras, {df['Label'].nunique()} clases)")
        return df

    try:
        process_split(train_list, train_poses_dir, os.path.join(train_dir, "labels.csv"), "train")
        process_split(test_list, test_poses_dir, os.path.join(test_dir, "labels.csv"), "test")
    finally:
        holistic.close()

    print(f"\nListo. Train: {train_poses_dir}")
    print(f"       Test:  {test_poses_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera datos de entrenamiento (poses + labels.csv) desde la carpeta de videos."
    )
    parser.add_argument(
        "videos_dir",
        nargs="?",
        default=VIDEOS_DIR,
        help=f"Carpeta con subcarpetas por palabra. Por defecto: {VIDEOS_DIR!r} (constante VIDEOS_DIR)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Carpeta de salida (data/train/poses y labels.csv). Por defecto: train-data/extraction_output",
    )
    parser.add_argument(
        "--max-folders",
        type=int,
        default=MAX_FOLDERS if MAX_FOLDERS is not None else 0,
        help="Procesar solo las primeras N carpetas. 0 = todas. Por defecto usa la constante MAX_FOLDERS.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="No reprocesar videos que ya tengan .npy",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=TEST_SPLIT,
        help=f"Proporción de muestras para test (ej. 0.2 = 20%%). Por defecto: {TEST_SPLIT} (constante TEST_SPLIT).",
    )
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    args = parser.parse_args()

    # Por defecto salida en train-data/extraction_output para usar desde raíz del repo
    if args.output is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(base, "extraction_output")

    # 0 = procesar todas las carpetas
    max_folders = None if args.max_folders <= 0 else args.max_folders
    run_extraction(
        videos_dir=args.videos_dir,
        output_dir=args.output,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        skip_existing=args.skip_existing,
        max_folders=max_folders,
        test_split=args.test_split,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
