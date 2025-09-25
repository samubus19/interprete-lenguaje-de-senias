import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import json
from pathlib import Path

class FrameExtractor:
    def __init__(self, target_frames=30):
        """
        Extractor de frames representativos de videos de LSA
        
        Args:
            target_frames (int): Número de frames a extraer por video
        """
        self.target_frames = target_frames
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_frames_from_video(self, video_path):
        """
        Extrae todos los frames de un video
        
        Args:
            video_path (str): Ruta al archivo de video
            
        Returns:
            list: Lista de frames (numpy arrays)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            print(f"Error: No se puede abrir el video {video_path}")
            return frames
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames
    
    def calculate_frame_features(self, frames):
        """
        Calcula características de cada frame usando MediaPipe Holistic
        
        Args:
            frames (list): Lista de frames
            
        Returns:
            tuple: (features, valid_frames) - características y frames válidos
        """
        features = []
        valid_frames = []
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            for i, frame in enumerate(frames):
                # Procesar frame con MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Extraer características
                frame_features = self._extract_holistic_features(results)
                
                # Solo incluir frames con detecciones válidas
                if frame_features is not None:
                    features.append(frame_features)
                    valid_frames.append((i, frame))
                    
        return np.array(features), valid_frames
    
    def _extract_holistic_features(self, results):
        """
        Extrae características de los landmarks de MediaPipe Holistic
        
        Args:
            results: Resultados de MediaPipe Holistic
            
        Returns:
            numpy.array: Vector de características o None si no hay detecciones
        """
        features = []
        
        # Landmarks de pose (33 puntos * 4 coordenadas = 132)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            features.extend([0.0] * 132)  # Rellenar con ceros si no hay detección
            
        # Landmarks de mano izquierda (21 puntos * 3 coordenadas = 63)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * 63)
            
        # Landmarks de mano derecha (21 puntos * 3 coordenadas = 63)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * 63)
            
        # Landmarks faciales (468 puntos * 3 coordenadas = 1404)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * 1404)
            
        # Solo retornar si hay al menos detección de manos o pose
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            return np.array(features)
        else:
            return None
    
    def select_representative_frames(self, features, valid_frames):
        """
        Selecciona frames representativos usando clustering
        
        Args:
            features (numpy.array): Características de los frames
            valid_frames (list): Lista de (índice, frame) válidos
            
        Returns:
            list: Lista de frames representativos
        """
        if len(features) == 0:
            print("No se encontraron frames válidos")
            return []
            
        # Si hay menos frames que el objetivo, devolver todos
        if len(features) <= self.target_frames:
            return [frame for _, frame in valid_frames]
            
        # Usar K-means para agrupar frames similares
        n_clusters = min(self.target_frames, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        representative_frames = []
        
        # Seleccionar el frame más cercano al centroide de cada cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_features = features[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Encontrar el frame más cercano al centro del cluster
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            
            representative_frames.append(valid_frames[closest_idx][1])
            
        return representative_frames
    
    def process_video(self, video_path, output_dir=None):
        """
        Procesa un video completo y extrae frames representativos
        
        Args:
            video_path (str): Ruta al video
            output_dir (str): Directorio de salida (opcional)
            
        Returns:
            list: Lista de frames representativos
        """
        print(f"Procesando video: {video_path}")
        
        # Extraer todos los frames
        frames = self.extract_frames_from_video(video_path)
        if not frames:
            return []
            
        print(f"Frames extraídos: {len(frames)}")
        
        # Calcular características
        features, valid_frames = self.calculate_frame_features(frames)
        print(f"Frames válidos con detecciones: {len(valid_frames)}")
        
        # Seleccionar frames representativos
        representative_frames = self.select_representative_frames(features, valid_frames)
        print(f"Frames representativos seleccionados: {len(representative_frames)}")
        
        # Guardar frames si se especifica directorio de salida
        if output_dir:
            self.save_frames(representative_frames, output_dir, video_path)
            
        return representative_frames
    
    def save_frames(self, frames, output_dir, video_path):
        """
        Guarda los frames representativos en disco
        
        Args:
            frames (list): Lista de frames
            output_dir (str): Directorio de salida
            video_path (str): Ruta del video original (para nombrar)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            
        print(f"Frames guardados en: {output_dir}")
    
    def process_phrase_directory(self, phrase_dir, output_base_dir="processed_frames"):
        """
        Procesa todos los videos de una frase específica
        
        Args:
            phrase_dir (str): Directorio de la frase (ej: videos/HOLA_COMO_ESTAS)
            output_base_dir (str): Directorio base de salida
            
        Returns:
            dict: Información del procesamiento
        """
        phrase_name = os.path.basename(phrase_dir)
        output_dir = os.path.join(output_base_dir, phrase_name)
        
        video_files = [f for f in os.listdir(phrase_dir) if f.endswith('.avi')]
        
        processing_info = {
            'phrase': phrase_name,
            'total_videos': len(video_files),
            'processed_videos': 0,
            'total_frames': 0
        }
        
        for video_file in video_files:
            video_path = os.path.join(phrase_dir, video_file)
            video_output_dir = os.path.join(output_dir, Path(video_file).stem)
            
            frames = self.process_video(video_path, video_output_dir)
            
            if frames:
                processing_info['processed_videos'] += 1
                processing_info['total_frames'] += len(frames)
                
        return processing_info
    
    def process_all_phrases(self, videos_dir="videos", output_base_dir="processed_frames"):
        """
        Procesa todos los videos de todas las frases
        
        Args:
            videos_dir (str): Directorio base de videos
            output_base_dir (str): Directorio base de salida
            
        Returns:
            dict: Resumen del procesamiento
        """
        if not os.path.exists(videos_dir):
            print(f"Error: El directorio {videos_dir} no existe")
            return {}
            
        phrase_dirs = [d for d in os.listdir(videos_dir) 
                      if os.path.isdir(os.path.join(videos_dir, d))]
        
        summary = {
            'total_phrases': len(phrase_dirs),
            'phrases_processed': [],
            'total_videos': 0,
            'total_frames': 0
        }
        
        for phrase_dir in phrase_dirs:
            phrase_path = os.path.join(videos_dir, phrase_dir)
            print(f"\n=== Procesando frase: {phrase_dir} ===")
            
            info = self.process_phrase_directory(phrase_path, output_base_dir)
            summary['phrases_processed'].append(info)
            summary['total_videos'] += info['processed_videos']
            summary['total_frames'] += info['total_frames']
            
        # Guardar resumen
        summary_path = os.path.join(output_base_dir, 'processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        return summary


def main():
    """Función principal para ejecutar el extractor de frames"""
    extractor = FrameExtractor(target_frames=30)
    
    print("=== Extractor de Frames Representativos para LSA ===")
    print("Este script procesará todos los videos y extraerá 30 frames representativos de cada uno.")
    
    # Procesar todos los videos
    summary = extractor.process_all_phrases()
    
    if summary:
        print("\n=== RESUMEN DEL PROCESAMIENTO ===")
        print(f"Frases procesadas: {summary['total_phrases']}")
        print(f"Videos procesados: {summary['total_videos']}")
        print(f"Frames extraídos: {summary['total_frames']}")
        print(f"Promedio de frames por video: {summary['total_frames'] / max(summary['total_videos'], 1):.1f}")
        
        print("\nDetalles por frase:")
        for phrase_info in summary['phrases_processed']:
            print(f"- {phrase_info['phrase']}: {phrase_info['processed_videos']} videos, {phrase_info['total_frames']} frames")
            
        print(f"\nResultados guardados en: processed_frames/")
        print("Siguiente paso: Ejecutar landmark_extractor.py para extraer landmarks de estos frames")


if __name__ == "__main__":
    main()
