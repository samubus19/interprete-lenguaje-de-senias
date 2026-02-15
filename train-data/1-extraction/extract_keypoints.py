"""
Extrae keypoints (landmarks) de un video usando MediaPipe Holistic.

Formato de salida compatible con el Colab de entrenamiento:
- Array NumPy de forma (num_frames, 75, 3): 75 keypoints por frame,
  cada uno con (x, y, confidence). Los 75 son: 33 pose + 21 mano izquierda + 21 mano derecha.
- No se usan landmarks de rostro para coincidir con el modelo (75 keypoints).
"""

import os
import numpy as np
import cv2
import mediapipe as mp

# Número de keypoints que espera el Colab (33 pose + 21 mano izq + 21 mano der)
NUM_KEYPOINTS = 75
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21


def _landmarks_to_array(landmarks, default_confidence=1.0):
    """
    Convierte landmarks de MediaPipe a array (N, 3) con (x, y, confidence).
    MediaPipe usa coordenadas normalizadas [0, 1]. visibility existe en pose.
    """
    if landmarks is None:
        return None
    rows = []
    for lm in landmarks.landmark:
        conf = getattr(lm, "visibility", default_confidence)
        if conf is None:
            conf = default_confidence
        rows.append([lm.x, lm.y, conf])
    return np.array(rows, dtype=np.float32)


def _fill_or_trim(arr, target_len):
    """Devuelve un array de longitud target_len; rellena con ceros o recorta."""
    if arr is None:
        return np.zeros((target_len, 3), dtype=np.float32)
    n = len(arr)
    if n >= target_len:
        return arr[:target_len].astype(np.float32)
    out = np.zeros((target_len, 3), dtype=np.float32)
    out[:n] = arr
    return out


def frame_to_keypoints(holistic, frame_rgb):
    """
    Procesa un frame (RGB) y devuelve un vector de 75 keypoints (33 pose + 21 L + 21 R).
    Formato: (75, 3) con (x, y, confidence).
    """
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)
    frame_rgb.flags.writeable = True

    pose = _landmarks_to_array(results.pose_landmarks, default_confidence=1.0)
    left = _landmarks_to_array(results.left_hand_landmarks)
    right = _landmarks_to_array(results.right_hand_landmarks)

    pose_arr = _fill_or_trim(pose, POSE_LANDMARKS)
    left_arr = _fill_or_trim(left, HAND_LANDMARKS)   # cuando no hay mano, rellena 0 y confidence 0
    right_arr = _fill_or_trim(right, HAND_LANDMARKS)

    # Un solo vector de 75 keypoints por frame
    return np.concatenate([pose_arr, left_arr, right_arr], axis=0)


def extract_keypoints_from_video(
    video_path,
    output_path=None,
    holistic=None,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """
    Lee un video, extrae keypoints con MediaPipe Holistic y opcionalmente guarda .npy.

    Args:
        video_path: Ruta al archivo de video (.mp4, .avi, etc.).
        output_path: Si se indica, guarda el array en este path (.npy).
        holistic: Instancia de mp.solutions.holistic.Holistic o None para crear una.
        min_detection_confidence: Umbral de detección de MediaPipe.
        min_tracking_confidence: Umbral de seguimiento de MediaPipe.

    Returns:
        np.ndarray de forma (num_frames, 75, 3), dtype float32.
        Si el video no tiene frames válidos, devuelve array vacío (0, 75, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    own_holistic = False
    if holistic is None:
        own_holistic = True
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    frames_keypoints = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp = frame_to_keypoints(holistic, rgb)
            frames_keypoints.append(kp)
    finally:
        cap.release()
        if own_holistic:
            holistic.close()

    if not frames_keypoints:
        arr = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
    else:
        arr = np.stack(frames_keypoints, axis=0)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, arr)

    return arr


def main():
    """Ejemplo de uso: extraer keypoints de un video y guardar .npy."""
    import argparse
    parser = argparse.ArgumentParser(description="Extrae keypoints de un video (MediaPipe Holistic).")
    parser.add_argument("video", help="Ruta al video")
    parser.add_argument("-o", "--output", default=None, help="Ruta de salida .npy (por defecto: mismo nombre que el video con extensión .npy)")
    parser.add_argument("--min-detection", type=float, default=0.5, help="Min detection confidence")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="Min tracking confidence")
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = os.path.splitext(args.video)[0] + ".npy"

    arr = extract_keypoints_from_video(
        args.video,
        output_path=out,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
    )
    print(f"Video: {args.video} -> {arr.shape[0]} frames, guardado en {out}")


if __name__ == "__main__":
    main()
