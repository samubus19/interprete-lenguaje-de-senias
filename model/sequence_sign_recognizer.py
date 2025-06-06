import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import os

N_FRAMES = 20  # Debe coincidir con el recolector y el modelo

class SequenceSignRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # Cargar señas desde el archivo JSON
        with open("data/signs.json", "r", encoding="utf-8") as f:
            signs_data = json.load(f)
            self.signs = {i: sign["name"] for i, sign in enumerate(signs_data["signs"])}
        self.num_classes = len(self.signs)
        # Cargar modelo LSTM entrenado
        model_path = "models/sign_language_sequence_model.h5"
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print("Modelo LSTM de secuencias cargado correctamente.")
        else:
            self.model = None
            print("No se encontró el modelo LSTM de secuencias.")
        # Buffer de secuencia
        self.sequence_buffer = []
        self.last_prediction = None

    def preprocess_landmarks(self, landmarks):
        if not landmarks:
            return None
        data = []
        for landmark in landmarks.landmark:
            data.extend([landmark.x, landmark.y])
        return np.array(data)

    def update_buffer(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            processed = self.preprocess_landmarks(hand_landmarks)
            if processed is not None:
                self.sequence_buffer.append(processed)
                if len(self.sequence_buffer) > N_FRAMES:
                    self.sequence_buffer.pop(0)
        # Si no hay mano, no agregamos nada

    def predict(self):
        if self.model is None or len(self.sequence_buffer) < N_FRAMES:
            return None
        sequence = np.array(self.sequence_buffer[-N_FRAMES:])
        sequence = np.expand_dims(sequence, axis=0)  # (1, N_FRAMES, 42)
        prediction = self.model.predict(sequence, verbose=0)
        sign_idx = np.argmax(prediction[0])
        confidence = prediction[0][sign_idx]
        if confidence > 0.3:
            return {'sign': self.signs[sign_idx], 'confidence': float(confidence)}
        return None

    def reset_buffer(self):
        self.sequence_buffer = [] 