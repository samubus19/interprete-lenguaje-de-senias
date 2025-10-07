import cv2
import os
import numpy as np
import mediapipe as mp
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

class SequenceDataProcessor:
    def __init__(self, max_sequence_length=30):
        """
        Procesador de datos para secuencias de LSA
        
        Args:
            max_sequence_length (int): Longitud máxima de secuencia
        """
        self.max_sequence_length = max_sequence_length
        self.mp_holistic = mp.solutions.holistic
        
        # Configuración de MediaPipe
        self.holistic_config = {
            'static_image_mode': True,
            'model_complexity': 1,
            'enable_segmentation': False,
            'refine_face_landmarks': False,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
    
    def extract_keypoints_from_image(self, image_path):
        """
        Extrae keypoints de una imagen usando MediaPipe Holistic
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            numpy.array: Vector de keypoints (1662 características) o None
        """
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se puede leer la imagen {image_path}")
            return None
            
        # Convertir a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        with self.mp_holistic.Holistic(**self.holistic_config) as holistic:
            results = holistic.process(rgb_image)
            
        return self._extract_keypoints(results)
    
    def _extract_keypoints(self, results):
        """
        Extrae keypoints de los resultados de MediaPipe
        
        Args:
            results: Resultados de MediaPipe Holistic
            
        Returns:
            numpy.array: Vector de keypoints
        """
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
        
        return np.array(keypoints)
    
    def process_video_sequence(self, video_frames_dir):
        """
        Procesa una secuencia de frames de un video
        
        Args:
            video_frames_dir (str): Directorio con frames del video
            
        Returns:
            numpy.array: Secuencia de keypoints (max_sequence_length, 1662)
        """
        # Obtener archivos de frames ordenados
        frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        
        if len(frame_files) == 0:
            print(f"No se encontraron frames en {video_frames_dir}")
            return None
        
        sequence_keypoints = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(video_frames_dir, frame_file)
            keypoints = self.extract_keypoints_from_image(frame_path)
            
            if keypoints is not None:
                sequence_keypoints.append(keypoints)
        
        if len(sequence_keypoints) == 0:
            print(f"No se pudieron extraer keypoints de {video_frames_dir}")
            return None
        
        # Convertir a numpy array
        sequence_keypoints = np.array(sequence_keypoints)
        
        # Aplicar padding o truncar para longitud fija
        if len(sequence_keypoints) > self.max_sequence_length:
            # Truncar
            sequence_keypoints = sequence_keypoints[:self.max_sequence_length]
        elif len(sequence_keypoints) < self.max_sequence_length:
            # Padding con ceros
            padding_length = self.max_sequence_length - len(sequence_keypoints)
            padding = np.zeros((padding_length, sequence_keypoints.shape[1]))
            sequence_keypoints = np.vstack([sequence_keypoints, padding])
        
        return sequence_keypoints
    
    def process_all_data(self, processed_frames_dir="processed_frames", output_file="sequence_dataset.pkl"):
        """
        Procesa todos los datos y crea dataset para entrenamiento
        
        Args:
            processed_frames_dir (str): Directorio con frames procesados
            output_file (str): Archivo de salida
            
        Returns:
            dict: Dataset procesado
        """
        if not os.path.exists(processed_frames_dir):
            print(f"Error: No existe el directorio {processed_frames_dir}")
            return None
        
        X = []  # Secuencias de keypoints
        y = []  # Labels
        class_names = []
        
        # Obtener todas las clases (frases)
        phrase_dirs = [d for d in os.listdir(processed_frames_dir) 
                      if os.path.isdir(os.path.join(processed_frames_dir, d))]
        
        phrase_dirs = sorted(phrase_dirs)
        print(f"Frases encontradas: {phrase_dirs}")
        
        for class_id, phrase_name in enumerate(phrase_dirs):
            class_names.append(phrase_name)
            phrase_path = os.path.join(processed_frames_dir, phrase_name)
            
            # Obtener todos los videos de esta frase
            video_dirs = [d for d in os.listdir(phrase_path) 
                         if os.path.isdir(os.path.join(phrase_path, d))]
            
            print(f"Procesando '{phrase_name}': {len(video_dirs)} videos")
            
            for video_dir in video_dirs:
                video_path = os.path.join(phrase_path, video_dir)
                
                # Procesar secuencia de este video
                sequence = self.process_video_sequence(video_path)
                
                if sequence is not None:
                    X.append(sequence)
                    y.append(class_id)
                    print(f"  ✓ Video {video_dir}: {sequence.shape}")
                else:
                    print(f"  ✗ Error procesando video {video_dir}")
        
        if len(X) == 0:
            print("Error: No se pudieron procesar datos")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset creado:")
        print(f"- Secuencias: {X.shape}")
        print(f"- Labels: {y.shape}")
        print(f"- Clases: {len(class_names)}")
        
        # Crear dataset
        dataset = {
            'X': X,
            'y': y,
            'class_names': class_names,
            'max_sequence_length': self.max_sequence_length,
            'keypoints_length': X.shape[2],
            'num_classes': len(class_names)
        }
        
        # Guardar dataset
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset guardado en: {output_file}")
        
        return dataset
    
    def create_train_test_split(self, dataset, test_size=0.2, random_state=42):
        """
        Divide el dataset en entrenamiento y prueba
        
        Args:
            dataset (dict): Dataset procesado
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla aleatoria
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X = dataset['X']
        y = dataset['y']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main():
    """Función principal para procesar datos"""
    print("=== PROCESADOR DE SECUENCIAS LSA ===")
    
    processor = SequenceDataProcessor(max_sequence_length=30)
    
    # Procesar todos los datos
    dataset = processor.process_all_data(
        processed_frames_dir="processed_frames",
        output_file="models/sequence_dataset.pkl"
    )
    
    if dataset:
        print(f"\n=== RESUMEN DEL DATASET ===")
        print(f"Forma de X: {dataset['X'].shape}")
        print(f"Forma de y: {dataset['y'].shape}")
        print(f"Clases: {dataset['class_names']}")
        print(f"Longitud de secuencia: {dataset['max_sequence_length']}")
        print(f"Longitud de keypoints: {dataset['keypoints_length']}")
        
        # Crear división train/test
        X_train, X_test, y_train, y_test = processor.create_train_test_split(dataset)
        print(f"\nDivisión train/test:")
        print(f"- Entrenamiento: {X_train.shape[0]} secuencias")
        print(f"- Prueba: {X_test.shape[0]} secuencias")
        
        print(f"\n✅ Datos listos para entrenar modelo LSTM")
        print(f"Siguiente paso: python train_lstm_model.py")

if __name__ == "__main__":
    main()
