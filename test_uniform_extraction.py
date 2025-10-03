import os
from frame_extractor import FrameExtractor
from pathlib import Path

def test_uniform_extraction():
    """
    Prueba rápida del nuevo método de extracción uniforme
    """
    print("=== PRUEBA DE EXTRACCIÓN UNIFORME ===")
    
    # Buscar un video de ejemplo
    videos_dir = "videos"
    if not os.path.exists(videos_dir):
        print(f"Error: No se encuentra el directorio {videos_dir}")
        return
    
    # Encontrar el primer video disponible
    video_path = None
    for phrase_dir in os.listdir(videos_dir):
        phrase_path = os.path.join(videos_dir, phrase_dir)
        if os.path.isdir(phrase_path):
            for video_file in os.listdir(phrase_path):
                if video_file.endswith('.avi'):
                    video_path = os.path.join(phrase_path, video_file)
                    break
            if video_path:
                break
    
    if not video_path:
        print("No se encontraron videos para probar")
        return
    
    print(f"Probando con video: {video_path}")
    
    # Crear extractor y procesar
    extractor = FrameExtractor(target_frames=30)
    
    # Extraer todos los frames primero para ver el total
    all_frames = extractor.extract_frames_from_video(video_path)
    print(f"Total de frames en el video: {len(all_frames)}")
    
    # Ahora usar el método uniforme
    selected_frames = extractor.select_uniform_frames(all_frames)
    print(f"Frames seleccionados: {len(selected_frames)}")
    
    # Mostrar los índices que se seleccionarían
    if len(all_frames) > 0:
        step = len(all_frames) / 30
        indices = []
        for i in range(30):
            frame_index = int(i * step)
            frame_index = min(frame_index, len(all_frames) - 1)
            indices.append(frame_index)
        
        print(f"Índices seleccionados: {indices}")
        print(f"Paso calculado: {step:.2f}")
        
        # Verificar que no hay duplicados
        unique_indices = set(indices)
        if len(unique_indices) != len(indices):
            print(f"WARNING: Hay índices duplicados!")
        else:
            print("✓ No hay índices duplicados")
    
    # Probar con el método completo
    print("\n--- Probando método completo ---")
    output_dir = "test_uniform_frames"
    frames = extractor.process_video(video_path, output_dir)
    
    if frames:
        print(f"✓ Extracción exitosa: {len(frames)} frames")
        print(f"Frames guardados en: {output_dir}")
    else:
        print("✗ Error en la extracción")

if __name__ == "__main__":
    test_uniform_extraction()
