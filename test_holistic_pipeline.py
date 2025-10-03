#!/usr/bin/env python3
"""
Script de prueba para el nuevo pipeline con modelo holístico
"""

import os
import sys

def test_holistic_pipeline():
    """Prueba el pipeline con el nuevo modelo holístico"""
    print("=== PRUEBA DEL PIPELINE HOLÍSTICO ===")
    
    # Verificar que existen los archivos necesarios
    required_files = [
        'frame_extractor.py',
        'landmark_extractor.py', 
        'train_holistic_model.py',
        'lsa_pipeline.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    
    # Verificar directorio de videos
    if not os.path.exists('videos'):
        print("⚠️  Directorio 'videos' no existe")
        print("   Ejecuta video_collector.py primero para grabar videos")
        return False
    
    # Contar videos disponibles
    video_count = 0
    phrase_count = 0
    
    for phrase_dir in os.listdir('videos'):
        phrase_path = os.path.join('videos', phrase_dir)
        if os.path.isdir(phrase_path):
            phrase_count += 1
            video_files = [f for f in os.listdir(phrase_path) if f.endswith('.avi')]
            video_count += len(video_files)
    
    print(f"📹 Videos encontrados: {video_count} videos en {phrase_count} frases")
    
    if video_count == 0:
        print("❌ No se encontraron videos para procesar")
        return False
    
    print("\n=== PASOS DEL PIPELINE ===")
    print("1. Extracción de frames (uniforme)")
    print("2. Extracción de landmarks (holístico: 378 características)")
    print("3. Entrenamiento de modelo básico (red densa)")
    print("4. Configuración de convertidor de glosas")
    
    print(f"\n=== COMANDOS PARA EJECUTAR ===")
    print("Para ejecutar paso a paso:")
    print("  python lsa_pipeline.py --step frames")
    print("  python lsa_pipeline.py --step landmarks") 
    print("  python lsa_pipeline.py --step train")
    print("  python lsa_pipeline.py --step gloss")
    print("")
    print("Para ejecutar todo el pipeline:")
    print("  python lsa_pipeline.py")
    print("")
    print("Para probar solo el entrenamiento:")
    print("  python train_holistic_model.py")
    
    return True

def main():
    success = test_holistic_pipeline()
    
    if success:
        print(f"\n✅ Pipeline holístico listo para usar")
        
        # Preguntar si quiere ejecutar un paso
        response = input(f"\n¿Quieres ejecutar el paso de extracción de frames ahora? (y/n): ")
        if response.lower() in ['y', 'yes', 's', 'si']:
            print("Ejecutando extracción de frames...")
            os.system("python lsa_pipeline.py --step frames")
    else:
        print(f"\n❌ Hay problemas que resolver antes de usar el pipeline")

if __name__ == "__main__":
    main()
