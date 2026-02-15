"""
Procesa la carpeta de videos (videos/PALABRA/1.mp4, 2.mp4, ...) y genera la estructura
que espera el Colab de entrenamiento:

  output_dir/
  └── data/
      └── train/
          ├── labels.csv      (columnas: ID, Label)
          └── poses/
              ├── 0.npy
              ├── 1.npy
              └── ...

Cada .npy tiene forma (num_frames, 75, 3). Los IDs son consecutivos (0, 1, 2, ...) y
el Label es el nombre de la carpeta (ej. AGUA, CHAU).
"""

import os
import argparse
import pandas as pd
import mediapipe as mp

from extract_keypoints import extract_keypoints_from_video

# --- Constantes que podés editar ---
# Ruta donde están los videos (subcarpetas por palabra: AGUA, CHAU, etc.)
VIDEOS_DIR = "D:/Archivos/Documentos/Universidad/Trabajo Final/videos-20260120T020840Z-3-001/videos"
# Límite de carpetas a procesar (None = todas). Útil para probar con pocas (ej. 3).
MAX_FOLDERS = 3

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


def run_extraction(
    videos_dir,
    output_dir,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    skip_existing=False,
    max_folders=None,
):
    """
    Genera data/train/poses/*.npy y data/train/labels.csv en output_dir.

    Args:
        videos_dir: Carpeta raíz de videos (contiene subcarpetas AGUA, CHAU, ...).
        output_dir: Carpeta raíz de salida (se creará output_dir/data/train/poses y labels.csv ahí).
        min_detection_confidence: Pasado a MediaPipe.
        min_tracking_confidence: Pasado a MediaPipe.
        skip_existing: Si True, no reprocesar videos cuyo .npy ya exista.
        max_folders: Si es int, solo procesar las primeras N carpetas (None = todas).
    """
    train_dir = os.path.join(output_dir, "data", "train")
    poses_dir = os.path.join(train_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)

    video_list = find_video_files(videos_dir, max_folders=max_folders)
    if not video_list:
        print(f"No se encontraron videos en {videos_dir} (subcarpetas con {VIDEO_EXTENSIONS})")
        return
    if max_folders is not None:
        print(f"Procesando solo las primeras {max_folders} carpetas ({len(video_list)} videos).")

    labels_csv_path = os.path.join(train_dir, "labels.csv")
    rows = []
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    try:
        for sample_id, (label, video_path) in enumerate(video_list):
            npy_path = os.path.join(poses_dir, f"{sample_id}.npy")
            if skip_existing and os.path.isfile(npy_path):
                print(f"[skip] {video_path} -> {npy_path}")
                rows.append({"ID": sample_id, "Label": label})
                continue

            print(f"[{sample_id + 1}/{len(video_list)}] {label} - {os.path.basename(video_path)}")
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
    finally:
        holistic.close()

    df = pd.DataFrame(rows)
    df.to_csv(labels_csv_path, index=False)
    print(f"\nListo. Poses: {poses_dir}")
    print(f"Labels: {labels_csv_path} ({len(df)} muestras, {df['Label'].nunique()} clases)")


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
    )


if __name__ == "__main__":
    main()
