import cv2
import os
import mediapipe as mp
import numpy as np

# --- Constantes ---
# Ruta donde se guardarán los videos
VIDEOS_PATH = "videos" # Ahora crea una carpeta en el directorio actual llamada "videos"
# Formato de video
VIDEO_FORMAT = ".avi"
# Codec para el video
CODEC = cv2.VideoWriter_fourcc(*'XVID')
# Frames de espera antes de comenzar a grabar
WAIT_FRAMES = 15

# Mensajes de estado
STATUS_READY = "Listo para capturar"
STATUS_PREPARING = "Preparando..."
STATUS_RECORDING = "Capturando..."

# --- Inicialización de MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Procesa la imagen y retorna las detecciones."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def has_hands(results):
    """Verifica si se detectaron manos en los resultados."""
    return results.left_hand_landmarks or results.right_hand_landmarks

def draw_styled_landmarks(image, results):
    """Dibuja los landmarks de las manos con estilo personalizado."""
    # Dibuja landmarks de la mano derecha
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=2)
                                 )
    # Dibuja landmarks de la mano izquierda
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=2)
                                 )

def start_recording(phrase_path, frame):
    """Inicia una nueva grabación y retorna el video_writer."""
    # Crear la carpeta para la frase si no existe
    os.makedirs(phrase_path, exist_ok=True)
    
    # Determinar el siguiente número de secuencia para rellenar huecos
    existing_videos = [f for f in os.listdir(phrase_path) if f.endswith(VIDEO_FORMAT)]
    numeric_videos_set = {int(os.path.splitext(v)[0]) for v in existing_videos if os.path.splitext(v)[0].isdigit()}
    
    sequence_count = 1
    while sequence_count in numeric_videos_set:
        sequence_count += 1

    video_path = os.path.join(phrase_path, f"{sequence_count}{VIDEO_FORMAT}")
    video_writer = cv2.VideoWriter(video_path, CODEC, 20.0, (frame.shape[1], frame.shape[0]))
    print(f"Comenzando a grabar: {video_path}")
    return video_writer

def handle_recording_logic(hands_detected, is_recording, status, wait_counter, video_writer, phrase_path, frame):
    """Maneja la lógica de grabación y retorna el nuevo estado."""
    if not is_recording:
        if hands_detected:
            if wait_counter < WAIT_FRAMES:
                status = f"{STATUS_PREPARING} ({wait_counter}/{WAIT_FRAMES})"
                wait_counter += 1
            else:
                is_recording = True
                status = STATUS_RECORDING
                video_writer = start_recording(phrase_path, frame)
        else:
            wait_counter = 0
            status = STATUS_READY
    
    else: # Está grabando
        if hands_detected:
            video_writer.write(frame)
        else:
            is_recording = False
            status = STATUS_READY
            wait_counter = 0
            if video_writer:
                video_writer.release()
                print("Grabación detenida.")
                video_writer = None
    
    return is_recording, status, wait_counter, video_writer

def draw_interface(image, phrase, status, phrase_path):
    """Dibuja la interfaz de usuario en la imagen."""
    # Definir color de fondo según el estado
    if STATUS_RECORDING in status:
        bg_color = (90, 200, 0)  # Verde
    elif STATUS_PREPARING in status:
        bg_color = (0, 255, 255) # Amarillo
    else: # STATUS_READY
        bg_color = (245, 117, 16) # Azul

    # Fondo para el texto
    cv2.rectangle(image, (0, 0), (640, 60), bg_color, -1)

    # Frase que se está grabando
    cv2.putText(image, f"Frase: {phrase}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if STATUS_PREPARING in status else (255, 255, 255), 1, cv2.LINE_AA)
    
    # Número de secuencias
    sequence_count_display = len([f for f in os.listdir(phrase_path) if f.endswith(VIDEO_FORMAT)]) if os.path.exists(phrase_path) else 0
    cv2.putText(image, f"Secuencias: {sequence_count_display}", (350, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if STATUS_PREPARING in status else (255, 255, 255), 1, cv2.LINE_AA)

    # Estado de la captura
    cv2.putText(image, f"Estado: {status}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if STATUS_PREPARING in status else (255, 255, 255), 1, cv2.LINE_AA)

def main():
    # Solicitar la frase a grabar
    # TODO: Luego estas frases o palabras pueden venir desde el diccionario o desde signs.json
    phrase = input("Introduce la palabra o frase a grabar: ")
    # Ruta de la carpeta para la frase
    phrase_path = os.path.join(VIDEOS_PATH, phrase.upper().replace(" ", "_"))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    # Estado de la grabación
    is_recording = False
    status = STATUS_READY
    wait_counter = 0
    video_writer = None
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame.")
                break

            # Detección con MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Lógica de grabación
            hands_detected = has_hands(results)
            is_recording, status, wait_counter, video_writer = handle_recording_logic(
                hands_detected, is_recording, status, wait_counter, video_writer, phrase_path, frame
            )

            # Dibujar interfaz
            draw_interface(image, phrase, status, phrase_path)

            cv2.imshow('Video Collector', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 