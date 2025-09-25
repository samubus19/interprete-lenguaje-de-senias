import cv2
import os
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
import pickle

class LandmarkExtractor:
    def __init__(self):
        """
        Extractor de landmarks usando MediaPipe Holistic para LSA
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuración de MediaPipe
        self.holistic_config = {
            'static_image_mode': True,
            'model_complexity': 2,
            'enable_segmentation': False,
            'refine_face_landmarks': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
    
    def extract_landmarks_from_image(self, image_path):
        """
        Extrae landmarks de una imagen usando MediaPipe Holistic
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            dict: Diccionario con landmarks extraídos
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
            
        # Extraer landmarks
        landmarks_data = {
            'image_path': image_path,
            'pose_landmarks': self._extract_pose_landmarks(results.pose_landmarks),
            'left_hand_landmarks': self._extract_hand_landmarks(results.left_hand_landmarks),
            'right_hand_landmarks': self._extract_hand_landmarks(results.right_hand_landmarks),
            'face_landmarks': self._extract_face_landmarks(results.face_landmarks),
            'has_detection': self._has_valid_detection(results)
        }
        
        return landmarks_data
    
    def _extract_pose_landmarks(self, pose_landmarks):
        """Extrae landmarks de pose"""
        if pose_landmarks is None:
            return None
            
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmarks
    
    def _extract_hand_landmarks(self, hand_landmarks):
        """Extrae landmarks de mano"""
        if hand_landmarks is None:
            return None
            
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        return landmarks
    
    def _extract_face_landmarks(self, face_landmarks):
        """Extrae landmarks faciales (solo puntos clave para eficiencia)"""
        if face_landmarks is None:
            return None
            
        # Solo extraer landmarks faciales clave (contorno, ojos, boca, nariz)
        key_face_indices = [
            # Contorno facial
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            # Ojos
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Boca
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Nariz
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151
        ]
        
        landmarks = []
        for i in key_face_indices:
            if i < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[i]
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'index': i
                })
        return landmarks
    
    def _has_valid_detection(self, results):
        """Verifica si hay detecciones válidas"""
        return (results.pose_landmarks is not None or 
                results.left_hand_landmarks is not None or 
                results.right_hand_landmarks is not None)
    
    def create_feature_vector(self, landmarks_data):
        """
        Crea un vector de características normalizado
        
        Args:
            landmarks_data (dict): Datos de landmarks
            
        Returns:
            numpy.array: Vector de características
        """
        features = []
        
        # Pose landmarks (33 puntos * 4 valores = 132)
        if landmarks_data['pose_landmarks']:
            for landmark in landmarks_data['pose_landmarks']:
                features.extend([landmark['x'], landmark['y'], landmark['z'], landmark['visibility']])
        else:
            features.extend([0.0] * 132)
            
        # Left hand landmarks (21 puntos * 3 valores = 63)
        if landmarks_data['left_hand_landmarks']:
            for landmark in landmarks_data['left_hand_landmarks']:
                features.extend([landmark['x'], landmark['y'], landmark['z']])
        else:
            features.extend([0.0] * 63)
            
        # Right hand landmarks (21 puntos * 3 valores = 63)
        if landmarks_data['right_hand_landmarks']:
            for landmark in landmarks_data['right_hand_landmarks']:
                features.extend([landmark['x'], landmark['y'], landmark['z']])
        else:
            features.extend([0.0] * 63)
            
        # Face landmarks clave (40 puntos * 3 valores = 120)
        if landmarks_data['face_landmarks']:
            for landmark in landmarks_data['face_landmarks']:
                features.extend([landmark['x'], landmark['y'], landmark['z']])
            # Rellenar si faltan landmarks
            while len(features) < 132 + 63 + 63 + 120:
                features.append(0.0)
        else:
            features.extend([0.0] * 120)
            
        return np.array(features[:378])  # Asegurar tamaño fijo
    
    def process_frame_sequence(self, frames_dir):
        """
        Procesa una secuencia de frames de un video
        
        Args:
            frames_dir (str): Directorio con frames de un video
            
        Returns:
            dict: Datos procesados de la secuencia
        """
        if not os.path.exists(frames_dir):
            print(f"Error: El directorio {frames_dir} no existe")
            return None
            
        # Obtener todos los frames ordenados
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            print(f"No se encontraron frames en {frames_dir}")
            return None
            
        sequence_data = {
            'video_path': frames_dir,
            'total_frames': len(frame_files),
            'landmarks_sequence': [],
            'feature_vectors': [],
            'valid_frames': 0
        }
        
        print(f"Procesando {len(frame_files)} frames de {os.path.basename(frames_dir)}")
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Extraer landmarks
            landmarks_data = self.extract_landmarks_from_image(frame_path)
            
            if landmarks_data and landmarks_data['has_detection']:
                # Crear vector de características
                feature_vector = self.create_feature_vector(landmarks_data)
                
                sequence_data['landmarks_sequence'].append(landmarks_data)
                sequence_data['feature_vectors'].append(feature_vector)
                sequence_data['valid_frames'] += 1
            else:
                # Agregar frame vacío para mantener secuencia temporal
                sequence_data['landmarks_sequence'].append(None)
                sequence_data['feature_vectors'].append(np.zeros(378))
                
        sequence_data['feature_vectors'] = np.array(sequence_data['feature_vectors'])
        
        return sequence_data
    
    def process_phrase_videos(self, phrase_frames_dir, output_dir):
        """
        Procesa todos los videos de una frase
        
        Args:
            phrase_frames_dir (str): Directorio con frames de la frase
            output_dir (str): Directorio de salida
            
        Returns:
            dict: Información del procesamiento
        """
        phrase_name = os.path.basename(phrase_frames_dir)
        phrase_output_dir = os.path.join(output_dir, phrase_name)
        os.makedirs(phrase_output_dir, exist_ok=True)
        
        # Obtener todos los directorios de videos
        video_dirs = [d for d in os.listdir(phrase_frames_dir) 
                     if os.path.isdir(os.path.join(phrase_frames_dir, d))]
        
        phrase_data = {
            'phrase': phrase_name,
            'total_videos': len(video_dirs),
            'processed_videos': 0,
            'sequences': []
        }
        
        for video_dir in video_dirs:
            video_frames_path = os.path.join(phrase_frames_dir, video_dir)
            
            # Procesar secuencia de frames
            sequence_data = self.process_frame_sequence(video_frames_path)
            
            if sequence_data and sequence_data['valid_frames'] > 0:
                # Guardar datos de la secuencia
                sequence_file = os.path.join(phrase_output_dir, f"{video_dir}_landmarks.pkl")
                with open(sequence_file, 'wb') as f:
                    pickle.dump(sequence_data, f)
                    
                phrase_data['sequences'].append({
                    'video_name': video_dir,
                    'total_frames': sequence_data['total_frames'],
                    'valid_frames': sequence_data['valid_frames'],
                    'file_path': sequence_file
                })
                
                phrase_data['processed_videos'] += 1
                
        # Guardar información de la frase
        phrase_info_file = os.path.join(phrase_output_dir, 'phrase_info.json')
        with open(phrase_info_file, 'w', encoding='utf-8') as f:
            json.dump(phrase_data, f, indent=4, ensure_ascii=False)
            
        return phrase_data
    
    def process_all_phrases(self, frames_base_dir="processed_frames", output_base_dir="landmarks_data"):
        """
        Procesa todas las frases y extrae landmarks
        
        Args:
            frames_base_dir (str): Directorio base con frames procesados
            output_base_dir (str): Directorio base de salida
            
        Returns:
            dict: Resumen completo del procesamiento
        """
        if not os.path.exists(frames_base_dir):
            print(f"Error: El directorio {frames_base_dir} no existe")
            print("Ejecuta frame_extractor.py primero")
            return {}
            
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Obtener todas las frases
        phrase_dirs = [d for d in os.listdir(frames_base_dir) 
                      if os.path.isdir(os.path.join(frames_base_dir, d)) and d != '__pycache__']
        
        summary = {
            'total_phrases': len(phrase_dirs),
            'phrases_data': [],
            'total_videos': 0,
            'total_sequences': 0,
            'feature_vector_shape': (378,)
        }
        
        for phrase_dir in phrase_dirs:
            phrase_frames_path = os.path.join(frames_base_dir, phrase_dir)
            print(f"\n=== Procesando landmarks para frase: {phrase_dir} ===")
            
            phrase_data = self.process_phrase_videos(phrase_frames_path, output_base_dir)
            
            summary['phrases_data'].append(phrase_data)
            summary['total_videos'] += phrase_data['processed_videos']
            summary['total_sequences'] += len(phrase_data['sequences'])
            
        # Guardar resumen general
        summary_file = os.path.join(output_base_dir, 'landmarks_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        return summary
    
    def create_unified_dataset(self, landmarks_base_dir="landmarks_data", output_file="dataset/lsa_sequence_dataset.pkl"):
        """
        Crea un dataset unificado con todas las secuencias
        
        Args:
            landmarks_base_dir (str): Directorio con datos de landmarks
            output_file (str): Archivo de salida del dataset
            
        Returns:
            dict: Información del dataset creado
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        dataset = {
            'sequences': [],
            'labels': [],
            'phrase_to_id': {},
            'id_to_phrase': {},
            'metadata': {
                'total_sequences': 0,
                'phrases': [],
                'feature_vector_shape': (378,),
                'max_sequence_length': 0
            }
        }
        
        phrase_id = 0
        
        # Procesar cada frase
        for phrase_dir in os.listdir(landmarks_base_dir):
            phrase_path = os.path.join(landmarks_base_dir, phrase_dir)
            
            if not os.path.isdir(phrase_path):
                continue
                
            # Cargar información de la frase
            phrase_info_file = os.path.join(phrase_path, 'phrase_info.json')
            if not os.path.exists(phrase_info_file):
                continue
                
            with open(phrase_info_file, 'r', encoding='utf-8') as f:
                phrase_info = json.load(f)
                
            phrase_name = phrase_info['phrase']
            dataset['phrase_to_id'][phrase_name] = phrase_id
            dataset['id_to_phrase'][phrase_id] = phrase_name
            dataset['metadata']['phrases'].append(phrase_name)
            
            # Cargar todas las secuencias de esta frase
            for sequence_info in phrase_info['sequences']:
                sequence_file = sequence_info['file_path']
                
                with open(sequence_file, 'rb') as f:
                    sequence_data = pickle.load(f)
                    
                feature_vectors = sequence_data['feature_vectors']
                
                # Actualizar longitud máxima de secuencia
                seq_length = len(feature_vectors)
                if seq_length > dataset['metadata']['max_sequence_length']:
                    dataset['metadata']['max_sequence_length'] = seq_length
                    
                dataset['sequences'].append(feature_vectors)
                dataset['labels'].append(phrase_id)
                dataset['metadata']['total_sequences'] += 1
                
            phrase_id += 1
            
        # Guardar dataset
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"\nDataset unificado creado: {output_file}")
        print(f"Total de secuencias: {dataset['metadata']['total_sequences']}")
        print(f"Total de frases: {len(dataset['metadata']['phrases'])}")
        print(f"Longitud máxima de secuencia: {dataset['metadata']['max_sequence_length']}")
        
        return dataset


def main():
    """Función principal"""
    extractor = LandmarkExtractor()
    
    print("=== Extractor de Landmarks para LSA ===")
    print("Este script procesará los frames extraídos y generará landmarks con MediaPipe Holistic")
    
    # Procesar todos los landmarks
    summary = extractor.process_all_phrases()
    
    if summary:
        print("\n=== RESUMEN DE LANDMARKS ===")
        print(f"Frases procesadas: {summary['total_phrases']}")
        print(f"Videos procesados: {summary['total_videos']}")
        print(f"Secuencias creadas: {summary['total_sequences']}")
        print(f"Forma del vector de características: {summary['feature_vector_shape']}")
        
        # Crear dataset unificado
        print("\n=== Creando dataset unificado ===")
        dataset = extractor.create_unified_dataset()
        
        print("\nProcesamiento completado!")
        print("Siguiente paso: Entrenar modelo de secuencias con dataset/lsa_sequence_dataset.pkl")


if __name__ == "__main__":
    main()
