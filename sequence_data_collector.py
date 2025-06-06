import os
import json
import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp

N_FRAMES = 20  # Número de frames por secuencia

class SequenceDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        
        # Cargar señas disponibles
        with open("data/signs.json", "r", encoding="utf-8") as f:
            signs_data = json.load(f)
            self.signs = {sign["id"]: sign["name"] for sign in signs_data["signs"]}

    def capture_sequence(self, n_frames=N_FRAMES):
        sequence = []
        frames_captured = 0
        while frames_captured < n_frames:
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                data = []
                for landmark in hand_landmarks.landmark:
                    data.extend([landmark.x, landmark.y])
                sequence.append(data)
                frames_captured += 1
                # Dibujar landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Frame {frames_captured}/{n_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Mano no detectada", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Secuencia de Seña", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para cancelar
                return None
        cv2.destroyAllWindows()
        return np.array(sequence)

    def run(self):
        os.makedirs('data/collected_sequences', exist_ok=True)
        print("\n=== Recolección de Secuencias de Lenguaje de Señas ===")
        print("Señas disponibles:")
        for sign_id, sign_name in self.signs.items():
            print(f"{sign_id}: {sign_name}")
        print("\nIngresa los IDs de las señas que quieres grabar (separados por coma)")
        print("Por ejemplo: 1,3,5")
        print("O presiona Enter para grabar todas las señas")
        selected_ids = input("\nIDs de señas a grabar: ").strip()
        if selected_ids:
            try:
                selected_ids = [int(id.strip()) for id in selected_ids.split(",")]
                signs_to_record = [id for id in selected_ids if id in self.signs]
            except ValueError:
                print("Error: Ingresa números válidos separados por coma")
                return
        else:
            signs_to_record = list(self.signs.keys())
        print("\nInstrucciones:")
        print(f"1. Para cada seña, muestra la seña en movimiento durante {N_FRAMES} frames")
        print("2. Presiona 'Enter' para comenzar a grabar cada secuencia")
        print("3. Presiona 'ESC' en la ventana de video para cancelar la secuencia actual\n")
        for sign_id in signs_to_record:
            sign_dir = os.path.join('data/collected_sequences', str(sign_id))
            os.makedirs(sign_dir, exist_ok=True)
            print(f"\n=== Grabando secuencias para la seña: {self.signs[sign_id]} (ID: {sign_id}) ===")
            n_samples = int(input("¿Cuántas secuencias quieres grabar para esta seña?: "))
            for i in range(n_samples):
                input(f"Presiona Enter para grabar la secuencia {i+1}/{n_samples}...")
                seq = self.capture_sequence()
                if seq is not None and len(seq) == N_FRAMES:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"seq_{timestamp}.npy"
                    filepath = os.path.join(sign_dir, filename)
                    np.save(filepath, seq)
                    print(f"Secuencia {i+1}/{n_samples} guardada.")
                else:
                    print("Secuencia cancelada o incompleta.")
        print("\nRecolección de secuencias completada!")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = SequenceDataCollector()
    collector.run() 