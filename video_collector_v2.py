import os
import cv2


VIDEOS_PATH = "videos"
VIDEO_FORMAT = ".mp4" # ".avi"
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30.0


def build_phrase_path(phrase: str) -> str:
    """Genera la ruta de almacenamiento para la frase."""
    safe_phrase = phrase.upper().strip().replace(" ", "_")
    return os.path.join(VIDEOS_PATH, safe_phrase)


def next_sequence_path(phrase_path: str) -> str:
    """Obtiene la próxima ruta disponible siguiendo el esquema numérico."""
    os.makedirs(phrase_path, exist_ok=True)
    existing = [
        f for f in os.listdir(phrase_path)
        if f.endswith(VIDEO_FORMAT) and os.path.splitext(f)[0].isdigit()
    ]
    used_numbers = {int(os.path.splitext(f)[0]) for f in existing}

    sequence = 1
    while sequence in used_numbers:
        sequence += 1

    filename = f"{sequence}{VIDEO_FORMAT}"
    return os.path.join(phrase_path, filename)


def select_fourcc(video_format: str) -> int:
    """Selecciona el fourcc apropiado según el formato solicitado."""
    ext = video_format.lower()
    if ext == ".mp4":
        return cv2.VideoWriter_fourcc(*'mp4v')
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*'MJPG')
    # Fallback genérico
    return cv2.VideoWriter_fourcc(*'mp4v')


def init_video_writer(width: int, height: int, fps: float, output_path: str) -> cv2.VideoWriter:
    """Inicializa el escritor con códec compatible manteniendo resolución/fps."""
    fps = fps if fps and fps > 1.0 else TARGET_FPS
    fourcc = select_fourcc(VIDEO_FORMAT)

    print(f"Propiedades finales → {width}x{height} @ {fps:.2f} FPS")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_recording_indicator(frame, is_recording: bool):
    """Dibuja un indicador minimalista de grabación SOLO en la vista previa (no en el video guardado)."""
    if not is_recording:
        return

    # Pequeño círculo rojo y texto "REC" en la esquina superior izquierda
    center = (25, 25)
    radius = 8
    cv2.circle(frame, center, radius, (0, 0, 255), -1)
    cv2.putText(
        frame,
        "REC",
        (45, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    phrase = input("Introduce la palabra o frase a grabar: ").strip()
    if not phrase:
        print("Error: La frase no puede estar vacía.")
        return

    phrase_path = build_phrase_path(phrase)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    # Intentar ajustar la cámara a la resolución deseada
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Leer el primer frame para conocer la resolución real disponible
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame inicial de la cámara.")
        cap.release()
        return

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    is_recording = False
    current_output_path = None

    print("Presiona ESPACIO para iniciar/detener la grabación.")
    print("Presiona 'q' para salir.")

    while True:
        # Escribir solo si estamos grabando
        if is_recording and writer:
            writer.write(frame)

        # Mostrar preview con indicador (sin modificar el frame que se guarda)
        display_frame = frame.copy()
        draw_recording_indicator(display_frame, is_recording)
        cv2.imshow("Video Collector V2", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Salir con 'q'
        if key == ord('q'):
            print("Saliendo por el usuario.")
            break

        # Alternar grabación con ESPACIO
        if key == ord(' '):
            if not is_recording:
                # Comenzar nueva grabación
                current_output_path = next_sequence_path(phrase_path)
                writer = init_video_writer(width, height, fps, current_output_path)
                if not writer.isOpened():
                    print("Error: No se pudo inicializar la escritura de video.")
                    writer = None
                    current_output_path = None
                else:
                    is_recording = True
                    print(f"[GRABANDO] {current_output_path}")
            else:
                # Detener grabación actual
                is_recording = False
                if writer:
                    writer.release()
                    print(f"[GUARDADO] {current_output_path}")
                    writer = None
                current_output_path = None

        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

    # Cerrar si quedaba una grabación abierta
    if writer:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")


if __name__ == "__main__":
    main()

