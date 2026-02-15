"""Funciones auxiliares para el pipeline de extracción."""

import os

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video_file(path):
    """Indica si el archivo tiene una extensión de video reconocida."""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def get_videos_base_dir_from_here():
    """
    Devuelve la ruta a la carpeta 'videos' asumiendo que está en la raíz del repo.
    Útil cuando se ejecuta desde train-data/1-extraction.
    """
    # 1-extraction -> train-data -> repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = os.path.dirname(script_dir)
    repo_root = os.path.dirname(train_data)
    return os.path.join(repo_root, "videos")
