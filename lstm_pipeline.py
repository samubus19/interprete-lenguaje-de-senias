#!/usr/bin/env python3
"""
Pipeline LSTM para el traductor de LSA (Lengua de Señas Argentina)
Ejecuta el pipeline completo desde extracción de frames hasta aplicación lista para usar
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

class LSTMPipeline:
    def __init__(self):
        """
        Pipeline LSTM para LSA
        Ejecuta todos los pasos necesarios para entrenar y usar el modelo LSTM
        """
        self.setup_directories()
        
    def setup_directories(self):
        """Crea directorios necesarios"""
        dirs_to_create = [
            'processed_frames',
            'models',
            'results'
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_extract_frames(self):
        """Paso 1: Extracción de frames representativos de videos"""
        print("=== PASO 1: EXTRACCIÓN DE FRAMES ===")
        
        # Verificar que existe el directorio de videos
        if not os.path.exists('videos'):
            print("❌ Error: Directorio 'videos' no existe")
            print("   Ejecuta primero: python video_collector.py")
            return False
        
        # Verificar que hay videos
        video_count = 0
        for root, dirs, files in os.walk('videos'):
            video_count += len([f for f in files if f.endswith('.avi')])
        
        if video_count == 0:
            print("❌ Error: No se encontraron videos (.avi) en el directorio 'videos'")
            print("   Ejecuta primero: python video_collector.py")
            return False
        
        print(f"📹 Encontrados {video_count} videos para procesar")
        
        # Ejecutar extracción de frames
        cmd = [sys.executable, 'frame_extractor.py']
        
        try:
            print("🔄 Ejecutando extracción de frames...")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Frames extraídos exitosamente")
                print("   Frames guardados en: processed_frames/")
                return True
            else:
                print("❌ Error en la extracción de frames")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando frame_extractor.py: {e}")
            return False
    
    def step2_process_sequences(self):
        """Paso 2: Procesamiento de secuencias usando sequence_data_processor.py"""
        print("\n=== PASO 2: PROCESAMIENTO DE SECUENCIAS ===")
        
        # Verificar que existen los frames procesados
        if not os.path.exists('processed_frames'):
            print("❌ Error: Directorio 'processed_frames' no existe")
            print("   Ejecuta primero el paso 1: extracción de frames")
            return False
        
        # Verificar que hay frames
        frame_count = 0
        for root, dirs, files in os.walk('processed_frames'):
            frame_count += len([f for f in files if f.endswith('.jpg')])
        
        if frame_count == 0:
            print("❌ Error: No se encontraron frames procesados")
            print("   Ejecuta primero el paso 1: extracción de frames")
            return False
        
        print(f"🖼️  Encontrados {frame_count} frames para procesar")
        
        # Ejecutar procesamiento de secuencias
        cmd = [sys.executable, 'sequence_data_processor.py']
        
        try:
            print("🔄 Ejecutando procesamiento de secuencias...")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Secuencias procesadas exitosamente")
                print("   Dataset guardado en: models/sequence_dataset.pkl")
                return True
            else:
                print("❌ Error en el procesamiento de secuencias")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando sequence_data_processor.py: {e}")
            return False
    
    def step3_data_augmentation(self):
        """Paso 3: Data augmentation usando data_augmentation.py"""
        print("\n=== PASO 3: DATA AUGMENTATION ===")
        
        # Verificar que existe el dataset de secuencias
        dataset_path = 'models/sequence_dataset.pkl'
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset '{dataset_path}' no existe")
            print("   Ejecuta primero el paso 2: procesamiento de secuencias")
            return False
        
        print("📊 Dataset original encontrado")
        
        # Ejecutar data augmentation
        cmd = [sys.executable, 'data_augmentation.py']
        
        try:
            print("🔄 Ejecutando data augmentation...")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Data augmentation completado exitosamente")
                print("   Dataset aumentado guardado en: models/sequence_dataset_augmented.pkl")
                return True
            else:
                print("❌ Error en data augmentation")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando data_augmentation.py: {e}")
            return False
    
    def step4_train_lstm_model(self):
        """Paso 4: Entrenamiento LSTM usando train_lstm_model.py"""
        print("\n=== PASO 4: ENTRENAMIENTO LSTM ===")
        
        # Verificar que existe el dataset aumentado
        augmented_dataset_path = 'models/sequence_dataset_augmented.pkl'
        original_dataset_path = 'models/sequence_dataset.pkl'
        
        dataset_to_use = None
        if os.path.exists(augmented_dataset_path):
            dataset_to_use = augmented_dataset_path
            print("📊 Usando dataset aumentado para entrenamiento")
        elif os.path.exists(original_dataset_path):
            dataset_to_use = original_dataset_path
            print("📊 Usando dataset original para entrenamiento")
            print("   ⚠️  Recomendación: Ejecuta data augmentation para mejores resultados")
        else:
            print("❌ Error: No se encontró ningún dataset")
            print("   Ejecuta primero los pasos 2 y 3")
            return False
        
        # Ejecutar entrenamiento LSTM
        cmd = [sys.executable, 'train_lstm_model.py']
        
        try:
            print("🔄 Ejecutando entrenamiento LSTM...")
            print("   ⏱️  Esto puede tomar varios minutos...")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Modelo LSTM entrenado exitosamente")
                print("   Modelo guardado en: models/lsa_lstm_model.h5")
                print("   Métricas guardadas en: models/lstm_training_metrics.json")
                return True
            else:
                print("❌ Error en el entrenamiento LSTM")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando train_lstm_model.py: {e}")
            return False
    
    def step5_ready_to_use(self):
        """Paso 5: Aplicación lista para usar con app_holistic.py"""
        print("\n=== PASO 5: APLICACIÓN LISTA ===")
        
        # Verificar que existe el modelo entrenado
        model_path = 'models/lsa_lstm_model.h5'
        if not os.path.exists(model_path):
            print(f"❌ Error: Modelo LSTM '{model_path}' no existe")
            print("   Ejecuta primero el paso 4: entrenamiento LSTM")
            return False
        
        # Verificar que existe el reconocedor LSTM
        lstm_recognizer_path = 'model/lstm_sign_recognizer.py'
        if not os.path.exists(lstm_recognizer_path):
            print(f"❌ Error: Reconocedor LSTM '{lstm_recognizer_path}' no existe")
            print("   Asegúrate de que el archivo lstm_sign_recognizer.py esté en model/")
            return False
        
        # Verificar que existe la aplicación
        app_path = 'app_holistic.py'
        if not os.path.exists(app_path):
            print(f"❌ Error: Aplicación '{app_path}' no existe")
            return False
        
        print("✅ Todos los componentes necesarios están presentes:")
        print(f"   📁 Modelo LSTM: {model_path}")
        print(f"   📁 Reconocedor: {lstm_recognizer_path}")
        print(f"   📁 Aplicación: {app_path}")
        
        # Mostrar instrucciones de uso
        print("\n🎉 ¡APLICACIÓN LISTA PARA USAR!")
        print("\n📋 INSTRUCCIONES DE USO:")
        print("   1. Para iniciar la aplicación:")
        print("      python app_holistic.py")
        print("\n   2. En la aplicación:")
        print("      - Presiona '▶️ Iniciar Reconocimiento' para comenzar")
        print("      - Realiza señas frente a la cámara")
        print("      - Las predicciones aparecerán en tiempo real")
        print("      - Usa '🗑️ Limpiar' para borrar el texto acumulado")
        print("\n   3. Para mejores resultados:")
        print("      - Asegúrate de tener buena iluminación")
        print("      - Mantén las manos visibles en la cámara")
        print("      - Realiza las señas de forma clara y pausada")
        
        return True
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo LSTM"""
        print("=" * 60)
        print("🤖 PIPELINE LSTM PARA TRADUCTOR LSA")
        print("=" * 60)
        print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Extracción de frames", self.step1_extract_frames),
            ("Procesamiento de secuencias", self.step2_process_sequences),
            ("Data augmentation", self.step3_data_augmentation),
            ("Entrenamiento LSTM", self.step4_train_lstm_model),
            ("Aplicación lista", self.step5_ready_to_use)
        ]
        
        results = {}
        total_steps = len(steps)
        
        for i, (step_name, step_function) in enumerate(steps, 1):
            print(f"\n{'=' * 60}")
            print(f"📍 PASO {i}/{total_steps}: {step_name.upper()}")
            print(f"{'=' * 60}")
            
            success = step_function()
            results[step_name] = success
            
            if not success:
                print(f"\n❌ PIPELINE DETENIDO EN: {step_name}")
                print("   Revisa los errores arriba y corrige antes de continuar")
                break
            
            print(f"✅ PASO {i}/{total_steps} COMPLETADO")
        
        # Resumen final
        print(f"\n{'=' * 60}")
        print("📊 RESUMEN DEL PIPELINE")
        print(f"{'=' * 60}")
        
        completed_steps = 0
        for i, (step_name, success) in enumerate(results.items(), 1):
            status = "✅ COMPLETADO" if success else "❌ FALLÓ"
            print(f"   {i}. {step_name}: {status}")
            if success:
                completed_steps += 1
        
        print(f"\n📈 Progreso: {completed_steps}/{total_steps} pasos completados")
        
        if all(results.values()):
            print(f"\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
            print(f"🚀 Tu traductor LSA con modelo LSTM está listo para usar")
            print(f"\n💡 Próximos pasos:")
            print(f"   - Ejecuta: python app_holistic.py")
            print(f"   - ¡Prueba tu traductor de señas!")
        else:
            failed_steps = [name for name, success in results.items() if not success]
            print(f"\n⚠️  PIPELINE INCOMPLETO")
            print(f"   Pasos fallidos: {', '.join(failed_steps)}")
            print(f"   Revisa los errores y ejecuta nuevamente")
        
        print(f"\n⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        
        return all(results.values())

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline LSTM para traductor LSA')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Ejecutar solo un paso específico (1-5)')
    
    args = parser.parse_args()
    
    # Crear pipeline
    pipeline = LSTMPipeline()
    
    if args.step:
        # Ejecutar paso específico
        step_methods = {
            1: pipeline.step1_extract_frames,
            2: pipeline.step2_process_sequences,
            3: pipeline.step3_data_augmentation,
            4: pipeline.step4_train_lstm_model,
            5: pipeline.step5_ready_to_use
        }
        
        print(f"Ejecutando solo el paso {args.step}")
        success = step_methods[args.step]()
        sys.exit(0 if success else 1)
    else:
        # Ejecutar pipeline completo
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
