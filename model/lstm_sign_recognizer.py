import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import os
from collections import deque

class LSTMSignRecognizer:
    def __init__(self, model_path=None, metadata_path=None):
        """
        Reconocedor de señas LSA usando modelo LSTM
        
        Args:
            model_path (str): Ruta al modelo entrenado
            metadata_path (str): Ruta a los metadatos
        """
        # Configurar MediaPipe Holistic (optimizado para rendimiento)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Reducido de 1 a 0 para mejor rendimiento
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.6,  # Aumentado para reducir falsos positivos
            min_tracking_confidence=0.6   # Aumentado para mejor tracking
        )
        
        # Inicializar variables
        self.model = None
        self.class_names = []
        self.max_sequence_length = 30
        self.keypoints_length = 1662
        
        # Buffer para secuencias
        self.sequence_buffer = deque(maxlen=self.max_sequence_length)
        self.prediction_buffer = deque(maxlen=5)  # Para suavizar predicciones
        
        # Cargar modelo si se proporciona
        if model_path:
            self.load_model(model_path, metadata_path)
    
    def load_model(self, model_path="models/lsa_lstm_model.h5", metadata_path=None):
        """
        Carga el modelo LSTM entrenado
        
        Args:
            model_path (str): Ruta al modelo
            metadata_path (str): Ruta a los metadatos
        """
        if not os.path.exists(model_path):
            print(f"❌ No se encontró el modelo en: {model_path}")
            return False
        
        try:
            # Cargar modelo
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Modelo LSTM cargado desde: {model_path}")
            
            # Cargar metadatos
            if metadata_path is None:
                metadata_path = model_path.replace('.h5', '_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.class_names = metadata['class_names']
                self.max_sequence_length = metadata['max_sequence_length']
                self.keypoints_length = metadata['keypoints_length']
                
                # Actualizar buffer size
                self.sequence_buffer = deque(maxlen=self.max_sequence_length)
                
                print(f"✅ Metadatos cargados: {len(self.class_names)} clases")
                print(f"   Clases: {', '.join(self.class_names)}")
            else:
                print(f"⚠️ No se encontraron metadatos en: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def extract_keypoints(self, frame):
        """
        Extrae keypoints de un frame usando MediaPipe Holistic
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            numpy.array: Vector de keypoints (1662) o None
        """
        if frame is None:
            return None
        
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.holistic.process(rgb_frame)
        
        # Extraer keypoints
        keypoints = []
        
        # Pose landmarks (33 puntos * 4 coordenadas = 132)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            keypoints.extend([0.0] * 132)
        
        # Face landmarks (468 puntos * 3 coordenadas = 1404)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * 1404)
        
        # Left hand landmarks (21 puntos * 3 coordenadas = 63)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * 63)
        
        # Right hand landmarks (21 puntos * 3 coordenadas = 63)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * 63)
        
        return np.array(keypoints), results
    
    def update_sequence_buffer(self, frame):
        """
        Actualiza el buffer de secuencia con un nuevo frame
        
        Args:
            frame: Frame de video
            
        Returns:
            bool: True si se agregó keypoints válidos
        """
        keypoints, results = self.extract_keypoints(frame)
        
        if keypoints is not None:
            # Verificar si hay detecciones válidas
            if np.sum(np.abs(keypoints)) > 0.1:  # No todo ceros
                self.sequence_buffer.append(keypoints)
                return True
        
        # Si no hay detecciones válidas, agregar ceros
        zero_keypoints = np.zeros(self.keypoints_length)
        self.sequence_buffer.append(zero_keypoints)
        return False
    
    def update_sequence_buffer_optimized(self, frame):
        """
        Versión optimizada que retorna también los resultados MediaPipe
        para evitar doble procesamiento
        
        Args:
            frame: Frame de video
            
        Returns:
            tuple: (bool: True si se agregó keypoints válidos, MediaPipe results)
        """
        keypoints, results = self.extract_keypoints(frame)
        
        if keypoints is not None:
            # Verificar si hay detecciones válidas
            if np.sum(np.abs(keypoints)) > 0.1:  # No todo ceros
                self.sequence_buffer.append(keypoints)
                return True, results
        
        # Si no hay detecciones válidas, agregar ceros
        zero_keypoints = np.zeros(self.keypoints_length)
        self.sequence_buffer.append(zero_keypoints)
        return False, results
    
    def predict_sequence(self, threshold=0.7):
        """
        Predice la seña basada en la secuencia actual
        
        Args:
            threshold (float): Umbral de confianza mínimo
            
        Returns:
            dict: Predicción con seña y confianza, o None
        """
        if self.model is None:
            return None
        
        if len(self.sequence_buffer) < self.max_sequence_length:
            return None
        
        # Preparar secuencia para predicción
        sequence = np.array(list(self.sequence_buffer))
        
        # Verificar que la secuencia no esté completamente vacía
        if np.sum(np.abs(sequence)) < 1.0:
            return None
        
        # Expandir dimensiones para el modelo
        sequence_input = np.expand_dims(sequence, axis=0)
        
        # Hacer predicción
        prediction = self.model.predict(sequence_input, verbose=0)[0]
        
        # Obtener clase con mayor probabilidad
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        if confidence >= threshold:
            if self.class_names:
                sign_name = self.class_names[predicted_class]
            else:
                sign_name = f"Clase_{predicted_class}"
            
            result = {
                'sign': sign_name,
                'confidence': float(confidence),
                'class_id': int(predicted_class),
                'all_probabilities': {
                    (self.class_names[i] if self.class_names else f"Clase_{i}"): float(prob)
                    for i, prob in enumerate(prediction)
                }
            }
            
            # Agregar al buffer de predicciones para suavizar
            self.prediction_buffer.append(result)
            
            return self._get_smoothed_prediction()
        
        return None
    
    def _get_smoothed_prediction(self):
        """
        Suaviza las predicciones usando el buffer
        
        Returns:
            dict: Predicción suavizada
        """
        if not self.prediction_buffer:
            return None
        
        # Contar frecuencias de cada seña
        sign_counts = {}
        total_confidence = 0
        
        for pred in self.prediction_buffer:
            sign = pred['sign']
            if sign not in sign_counts:
                sign_counts[sign] = {'count': 0, 'confidence_sum': 0}
            
            sign_counts[sign]['count'] += 1
            sign_counts[sign]['confidence_sum'] += pred['confidence']
            total_confidence += pred['confidence']
        
        # Encontrar la seña más frecuente
        most_frequent = max(sign_counts.items(), key=lambda x: x[1]['count'])
        sign = most_frequent[0]
        avg_confidence = most_frequent[1]['confidence_sum'] / most_frequent[1]['count']
        
        return {
            'sign': sign,
            'confidence': avg_confidence,
            'buffer_count': most_frequent[1]['count'],
            'buffer_size': len(self.prediction_buffer)
        }
    
    def reset_buffers(self):
        """Limpia todos los buffers"""
        self.sequence_buffer.clear()
        self.prediction_buffer.clear()
    
    def get_landmarks_for_drawing(self, frame):
        """
        Obtiene landmarks para dibujar en la interfaz
        
        Args:
            frame: Frame de video
            
        Returns:
            MediaPipe results para dibujar
        """
        if frame is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb_frame)
    
    def is_model_loaded(self):
        """Verifica si el modelo está cargado"""
        return self.model is not None
    
    def get_model_info(self):
        """Obtiene información del modelo"""
        if not self.is_model_loaded():
            return "Modelo LSTM no cargado"
        
        info = f"Modelo LSTM LSA cargado\n"
        info += f"Clases: {len(self.class_names)}\n"
        info += f"Longitud de secuencia: {self.max_sequence_length}\n"
        info += f"Keypoints por frame: {self.keypoints_length}\n"
        
        if self.class_names:
            info += f"Señas: {', '.join(self.class_names[:3])}..."
        
        return info
    
    def get_buffer_status(self):
        """Obtiene el estado de los buffers"""
        return {
            'sequence_buffer_size': len(self.sequence_buffer),
            'sequence_buffer_max': self.max_sequence_length,
            'prediction_buffer_size': len(self.prediction_buffer),
            'is_ready': len(self.sequence_buffer) >= self.max_sequence_length
        }
