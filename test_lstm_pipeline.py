#!/usr/bin/env python3
"""
Script de prueba completo para el pipeline LSTM de LSA
"""

import os
import sys

def check_processed_frames():
    """Verifica que existan frames procesados"""
    print("=== VERIFICANDO FRAMES PROCESADOS ===")
    
    frames_dir = "processed_frames"
    if not os.path.exists(frames_dir):
        print(f"❌ No existe el directorio {frames_dir}")
        return False
    
    phrase_dirs = [d for d in os.listdir(frames_dir) 
                  if os.path.isdir(os.path.join(frames_dir, d))]
    
    if not phrase_dirs:
        print(f"❌ No hay frases en {frames_dir}")
        return False
    
    total_videos = 0
    for phrase_dir in phrase_dirs:
        phrase_path = os.path.join(frames_dir, phrase_dir)
        video_dirs = [d for d in os.listdir(phrase_path) 
                     if os.path.isdir(os.path.join(phrase_path, d))]
        total_videos += len(video_dirs)
        
        print(f"  📁 {phrase_dir}: {len(video_dirs)} videos")
    
    print(f"✅ Total: {len(phrase_dirs)} frases, {total_videos} videos")
    return True

def check_sequence_dataset():
    """Verifica si existe el dataset de secuencias"""
    print("\n=== VERIFICANDO DATASET DE SECUENCIAS ===")
    
    dataset_path = "models/sequence_dataset.pkl"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset encontrado: {dataset_path}")
        
        # Cargar y mostrar información
        try:
            import pickle
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            print(f"  📊 Secuencias: {dataset['X'].shape}")
            print(f"  🏷️ Labels: {dataset['y'].shape}")
            print(f"  📝 Clases: {dataset['class_names']}")
            print(f"  📏 Longitud secuencia: {dataset['max_sequence_length']}")
            print(f"  🔢 Keypoints por frame: {dataset['keypoints_length']}")
            
            return True
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return False
    else:
        print(f"❌ Dataset no encontrado: {dataset_path}")
        return False

def check_lstm_model():
    """Verifica si existe el modelo LSTM entrenado"""
    print("\n=== VERIFICANDO MODELO LSTM ===")
    
    model_path = "models/lsa_lstm_model.h5"
    metadata_path = "models/lsa_lstm_model_metadata.json"
    
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado: {model_path}")
        
        if os.path.exists(metadata_path):
            print(f"✅ Metadatos encontrados: {metadata_path}")
            
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"  🧠 Tipo: {metadata['model_type']}")
                print(f"  📝 Clases: {metadata['class_names']}")
                print(f"  📏 Secuencia: {metadata['max_sequence_length']}")
                print(f"  🔢 Keypoints: {metadata['keypoints_length']}")
                
                return True
            except Exception as e:
                print(f"❌ Error cargando metadatos: {e}")
                return False
        else:
            print(f"⚠️ Metadatos no encontrados: {metadata_path}")
            return True
    else:
        print(f"❌ Modelo no encontrado: {model_path}")
        return False

def test_lstm_recognizer():
    """Prueba el reconocedor LSTM"""
    print("\n=== PROBANDO RECONOCEDOR LSTM ===")
    
    try:
        from model.lstm_sign_recognizer import LSTMSignRecognizer
        
        recognizer = LSTMSignRecognizer()
        
        if recognizer.is_model_loaded():
            print("✅ Reconocedor LSTM inicializado correctamente")
            print(recognizer.get_model_info())
            
            # Probar con frame dummy
            import numpy as np
            import cv2
            
            # Crear frame de prueba (negro)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Actualizar buffer
            recognizer.update_sequence_buffer(dummy_frame)
            buffer_status = recognizer.get_buffer_status()
            
            print(f"  📊 Buffer status: {buffer_status}")
            
            return True
        else:
            print("❌ No se pudo cargar el modelo LSTM")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando reconocedor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando reconocedor: {e}")
        return False

def show_pipeline_steps():
    """Muestra los pasos del pipeline LSTM"""
    print("\n=== PASOS DEL PIPELINE LSTM ===")
    
    steps = [
        ("1. Procesar frames → secuencias", "python sequence_data_processor.py"),
        ("2. Entrenar modelo LSTM", "python train_lstm_model.py"),
        ("3. Probar aplicación", "python app_holistic.py"),
        ("4. Pruebas adicionales", "python test_lstm_model.py")
    ]
    
    for step_name, command in steps:
        print(f"  {step_name}")
        print(f"    💻 {command}")

def main():
    """Función principal de pruebas"""
    print("🤟 PRUEBAS DEL PIPELINE LSTM PARA LSA")
    print("=" * 60)
    
    # Lista de verificaciones
    checks = [
        ("Frames procesados", check_processed_frames),
        ("Dataset de secuencias", check_sequence_dataset),
        ("Modelo LSTM", check_lstm_model),
        ("Reconocedor LSTM", test_lstm_recognizer)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            success = check_func()
            results[check_name] = success
        except Exception as e:
            print(f"❌ Error en {check_name}: {e}")
            results[check_name] = False
    
    # Resumen
    print(f"\n{'='*60}")
    print("=== RESUMEN DE VERIFICACIONES ===")
    
    all_passed = True
    for check_name, success in results.items():
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"  {check_name}: {status}")
        if not success:
            all_passed = False
    
    # Recomendaciones
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("Tu pipeline LSTM está listo para usar.")
        print("\n🚀 Para usar la aplicación:")
        print("   python app_holistic.py")
    else:
        print("⚠️ ALGUNAS VERIFICACIONES FALLARON")
        print("\n🔧 Pasos para completar el pipeline:")
        
        if not results.get("Frames procesados", False):
            print("   1. Ejecuta: python lsa_pipeline.py --step frames")
        
        if not results.get("Dataset de secuencias", False):
            print("   2. Ejecuta: python sequence_data_processor.py")
        
        if not results.get("Modelo LSTM", False):
            print("   3. Ejecuta: python train_lstm_model.py")
    
    show_pipeline_steps()

if __name__ == "__main__":
    main()
