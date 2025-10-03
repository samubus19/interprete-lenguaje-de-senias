import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import os

class HolisticSignRecognizer:
    def __init__(self):
        """
        Reconocedor de señas usando el modelo holístico entrenado
        """
        # Configurar MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cargar modelo y metadatos
        self.model = None
        self.class_names = []
        self.phrase_to_id = {}
        self.id_to_phrase = {}
        
        self._load_model()
        
        # Buffer para suavizar predicciones
        self.prediction_buffer = []
        self.buffer_size = 5
        self.last_prediction = None
        
    def _load_model(self):
        """Carga el modelo holístico entrenado"""
        model_path = "models/holistic_sign_model.h5"
        metadata_path = "models/holistic_model_metadata.json"
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("✓ Modelo holístico cargado correctamente")
                
                # Cargar metadatos si existen
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    self.class_names = metadata['class_names']
                    self.phrase_to_id = metadata['phrase_to_id']
                    self.id_to_phrase = {v: k for k, v in self.phrase_to_id.items()}
                    
                    print(f"✓ Metadatos cargados: {len(self.class_names)} clases")
                else:
                    print("⚠️ No se encontraron metadatos, usando índices numéricos")
                    
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
                self.model = None
        else:
            print(f"❌ No se encontró el modelo en: {model_path}")
            print("   Ejecuta 'python train_holistic_model.py' primero")
    
    def extract_holistic_features(self, frame):
        """
        Extrae características holísticas de un frame
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            numpy.array: Vector de 378 características o None
        """
        if frame is None:
            return None
            
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.holistic.process(rgb_frame)
        
        # Extraer características
        features = []
        
        # Pose landmarks (33 puntos * 4 valores = 132)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            features.extend([0.0] * 132)
            
        # Left hand landmarks (21 puntos * 3 valores = 63)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * 63)
            
        # Right hand landmarks (21 puntos * 3 valores = 63)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * 63)
            
        # Face landmarks clave (40 puntos * 3 valores = 120)
        if results.face_landmarks:
            # Extraer solo landmarks clave de la cara
            key_face_indices = [
                # Contorno facial
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                # Ojos
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # Cejas
                46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 55, 65, 70, 63, 105, 66, 107, 55, 65, 70,
                # Nariz
                1, 2, 5, 4, 6, 168, 8, 9, 10, 151,
                # Boca
                0, 11, 12, 13, 14, 15, 16, 17, 18, 200
            ]
            
            face_features = []
            for i in key_face_indices[:40]:  # Limitar a 40 puntos
                if i < len(results.face_landmarks.landmark):
                    landmark = results.face_landmarks.landmark[i]
                    face_features.extend([landmark.x, landmark.y, landmark.z])
                else:
                    face_features.extend([0.0, 0.0, 0.0])
            
            features.extend(face_features)
        else:
            features.extend([0.0] * 120)
        
        # Asegurar tamaño exacto de 378
        features = features[:378]
        while len(features) < 378:
            features.append(0.0)
            
        return np.array(features)
    
    def predict(self, frame, threshold=0.6):
        """
        Predice la frase de LSA desde un frame
        
        Args:
            frame: Frame de video
            threshold: Umbral de confianza mínimo
            
        Returns:
            dict: Predicción con frase y confianza, o None
        """
        if self.model is None:
            return None
            
        # Extraer características
        features = self.extract_holistic_features(frame)
        if features is None:
            return None
        
        # Verificar si hay detecciones válidas (no todo ceros)
        if np.sum(np.abs(features)) < 0.1:
            return None
            
        # Hacer predicción
        features_input = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features_input, verbose=0)[0]
        
        # Obtener clase con mayor probabilidad
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Aplicar umbral
        if confidence >= threshold:
            # Obtener nombre de la frase
            if self.class_names:
                phrase = self.class_names[predicted_class]
            else:
                phrase = f"Clase_{predicted_class}"
            
            result = {
                'phrase': phrase,
                'confidence': float(confidence),
                'class_id': int(predicted_class)
            }
            
            # Agregar al buffer para suavizar
            self.prediction_buffer.append(result)
            if len(self.prediction_buffer) > self.buffer_size:
                self.prediction_buffer.pop(0)
            
            # Devolver predicción más frecuente en el buffer
            return self._get_smoothed_prediction()
        
        return None
    
    def _get_smoothed_prediction(self):
        """Suaviza las predicciones usando el buffer"""
        if not self.prediction_buffer:
            return None
            
        # Contar frecuencias de cada frase
        phrase_counts = {}
        total_confidence = 0
        
        for pred in self.prediction_buffer:
            phrase = pred['phrase']
            if phrase not in phrase_counts:
                phrase_counts[phrase] = {'count': 0, 'confidence_sum': 0}
            
            phrase_counts[phrase]['count'] += 1
            phrase_counts[phrase]['confidence_sum'] += pred['confidence']
            total_confidence += pred['confidence']
        
        # Encontrar la frase más frecuente
        most_frequent = max(phrase_counts.items(), key=lambda x: x[1]['count'])
        phrase = most_frequent[0]
        avg_confidence = most_frequent[1]['confidence_sum'] / most_frequent[1]['count']
        
        return {
            'phrase': phrase,
            'confidence': avg_confidence,
            'buffer_size': len(self.prediction_buffer)
        }
    
    def reset_buffer(self):
        """Limpia el buffer de predicciones"""
        self.prediction_buffer = []
        self.last_prediction = None
    
    def get_landmarks_for_drawing(self, frame):
        """
        Obtiene landmarks para dibujar en la interfaz
        
        Returns:
            dict: Resultados de MediaPipe para dibujar
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
            return "Modelo no cargado"
            
        info = f"Modelo holístico cargado\n"
        info += f"Clases: {len(self.class_names)}\n"
        if self.class_names:
            info += f"Frases: {', '.join(self.class_names[:3])}..."
        
        return info
