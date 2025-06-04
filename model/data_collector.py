import os
import json
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="data/collected"):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
        
        # Cargar señas disponibles
        with open("data/signs.json", "r", encoding="utf-8") as f:
            signs_data = json.load(f)
            self.signs = {sign["id"]: sign["name"] for sign in signs_data["signs"]}
        
    def capture_frame(self):
        """Captura un frame de la cámara"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def detect_landmarks(self, frame):
        """Detecta los landmarks de las manos en el frame"""
        # Convertir el frame a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar el frame con MediaPipe
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Retornar los landmarks de la primera mano detectada
        return results.multi_hand_landmarks[0]
    
    def preprocess_landmarks(self, landmarks):
        """Preprocesa los landmarks para el modelo"""
        if not landmarks:
            return None
            
        # Convertir landmarks a un array numpy
        data = []
        for landmark in landmarks.landmark:
            data.extend([landmark.x, landmark.y])
        
        return np.array(data)
    
    def __del__(self):
        """Liberar recursos al destruir el objeto"""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def collect_data(self, sign_id, num_samples=30):
        """Recolecta datos para una seña específica"""
        if sign_id not in self.signs:
            print(f"Error: Seña ID {sign_id} no encontrada")
            return
        
        sign_name = self.signs[sign_id]
        print(f"\nRecolectando datos para la seña: {sign_name}")
        print("Presiona 'q' para salir o 'c' para capturar")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Dibujar landmarks si se detectan manos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Mostrar información en la pantalla
            cv2.putText(
                frame,
                f"Seña: {sign_name} - Muestras: {samples_collected}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Recolección de Datos", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and results.multi_hand_landmarks:
                # Guardar los datos de la seña
                self._save_sample(sign_id, results.multi_hand_landmarks[0])
                samples_collected += 1
                print(f"Muestra {samples_collected} recolectada")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nRecolección completada para {sign_name}")
        print(f"Muestras recolectadas: {samples_collected}")
    
    def _save_sample(self, sign_id, landmarks):
        """Guarda una muestra de datos"""
        # Crear directorio para la seña si no existe
        sign_dir = os.path.join(self.data_dir, str(sign_id))
        os.makedirs(sign_dir, exist_ok=True)
        
        # Convertir landmarks a array numpy
        data = []
        for landmark in landmarks.landmark:
            data.extend([landmark.x, landmark.y])
        
        # Generar nombre de archivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"sample_{timestamp}.npy"
        
        # Guardar datos
        np.save(os.path.join(sign_dir, filename), np.array(data))
    
    def prepare_training_data(self):
        """Prepara los datos recolectados para entrenamiento"""
        X = []  # Características
        y = []  # Etiquetas
        
        # Recorrer todos los directorios de señas
        for sign_id in os.listdir(self.data_dir):
            sign_dir = os.path.join(self.data_dir, sign_id)
            if not os.path.isdir(sign_dir):
                continue
                
            # Cargar todas las muestras de esta seña
            for sample_file in os.listdir(sign_dir):
                if sample_file.endswith('.npy'):
                    sample_path = os.path.join(sign_dir, sample_file)
                    data = np.load(sample_path)
                    X.append(data)
                    y.append(int(sign_id))
        
        return np.array(X), np.array(y)
    
    def get_data_stats(self):
        """Obtiene estadísticas de los datos recolectados"""
        stats = {}
        for sign_id in os.listdir(self.data_dir):
            sign_dir = os.path.join(self.data_dir, sign_id)
            if os.path.isdir(sign_dir):
                num_samples = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
                stats[self.signs.get(int(sign_id), f"Seña {sign_id}")] = num_samples
        return stats 