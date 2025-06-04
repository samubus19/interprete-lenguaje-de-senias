import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import mediapipe as mp
import cv2
import json
import os

class SignRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # Cargar señas desde el archivo JSON
        with open("data/signs.json", "r", encoding="utf-8") as f:
            signs_data = json.load(f)
            self.signs = {i: sign["name"] for i, sign in enumerate(signs_data["signs"])}
        
        # Construir el modelo
        self.model = self._build_model()
        
        # Intentar cargar el modelo entrenado
        model_path = "models/sign_language_model.h5"
        if os.path.exists(model_path):
            try:
                self.load_model(model_path)
                print("Modelo entrenado cargado correctamente.")
            except Exception as e:
                print("No se pudo cargar el modelo entrenado:", e)
        else:
            print("No se encontró un modelo entrenado. Se usará un modelo vacío.")
        
    def _build_model(self):
        """Construye el modelo de reconocimiento de señas"""
        input_shape = (42,)  # 21 puntos de referencia x 2 coordenadas (x,y)
        
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.signs), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_landmarks(self, landmarks):
        """Preprocesa los landmarks de las manos para el modelo"""
        if not landmarks:
            return None
            
        # Convertir landmarks a un array numpy
        data = []
        for landmark in landmarks.landmark:
            data.extend([landmark.x, landmark.y])
        
        return np.array(data)
    
    def predict(self, frame):
        """Realiza la predicción de señas en un frame"""
        # Convertir el frame a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar el frame con MediaPipe
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Procesar cada mano detectada
        predictions = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocesar landmarks
            processed_data = self.preprocess_landmarks(hand_landmarks)
            if processed_data is None:
                continue
                
            # Realizar predicción
            prediction = self.model.predict(
                np.expand_dims(processed_data, axis=0),
                verbose=0
            )
            
            # Obtener la seña con mayor probabilidad
            sign_idx = np.argmax(prediction[0])
            confidence = prediction[0][sign_idx]
            
            if confidence > 0.3:  # Umbral de confianza más bajo
                predictions.append({
                    'sign': self.signs[sign_idx],
                    'confidence': float(confidence)
                })
        
        return predictions
    
    def train(self, data, labels):
        """Entrena el modelo con datos etiquetados"""
        self.model.fit(
            data,
            labels,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
    
    def save_model(self, path):
        """Guarda el modelo entrenado"""
        self.model.save(path)
    
    def load_model(self, path):
        """Carga un modelo pre-entrenado"""
        self.model = tf.keras.models.load_model(path) 