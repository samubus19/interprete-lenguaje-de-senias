#!/usr/bin/env python3
"""
Pipeline completo para el traductor de LSA (Lengua de Señas Argentina) a texto
Integra todos los componentes del sistema de dos etapas
"""

import os
import sys
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime

from frame_extractor import FrameExtractor
from landmark_extractor import LandmarkExtractor
from model.sequence_model import LSASequenceModel
from gloss_to_text import GlossToTextConverter

class LSAPipeline:
    def __init__(self, config_file=None):
        """
        Pipeline completo para LSA
        
        Args:
            config_file (str): Archivo de configuración (opcional)
        """
        self.config = self._load_config(config_file)
        self.setup_directories()
        
    def _load_config(self, config_file):
        """Carga configuración del pipeline"""
        default_config = {
            'videos_dir': 'videos',
            'frames_dir': 'processed_frames',
            'landmarks_dir': 'landmarks_data',
            'dataset_dir': 'dataset',
            'models_dir': 'models',
            'results_dir': 'results',
            'target_frames': 30,
            'model_type': 'lstm',
            'gloss_method': 'rules',
            'openai_api_key': None
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def setup_directories(self):
        """Crea directorios necesarios"""
        dirs_to_create = [
            self.config['frames_dir'],
            self.config['landmarks_dir'],
            self.config['dataset_dir'],
            self.config['models_dir'],
            self.config['results_dir']
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_extract_frames(self):
        """Paso 1: Extraer frames representativos de videos"""
        print("=== PASO 1: EXTRACCIÓN DE FRAMES ===")
        
        if not os.path.exists(self.config['videos_dir']):
            print(f"Error: Directorio de videos '{self.config['videos_dir']}' no existe")
            print("Ejecuta video_collector.py primero para grabar videos")
            return False
            
        extractor = FrameExtractor(target_frames=self.config['target_frames'])
        summary = extractor.process_all_phrases(
            videos_dir=self.config['videos_dir'],
            output_base_dir=self.config['frames_dir']
        )
        
        if summary and summary['total_frames'] > 0:
            print(f"✓ Frames extraídos exitosamente: {summary['total_frames']} frames de {summary['total_videos']} videos")
            return True
        else:
            print("✗ Error en la extracción de frames")
            return False
    
    def step2_extract_landmarks(self):
        """Paso 2: Extraer landmarks de frames"""
        print("\n=== PASO 2: EXTRACCIÓN DE LANDMARKS ===")
        
        if not os.path.exists(self.config['frames_dir']):
            print(f"Error: Directorio de frames '{self.config['frames_dir']}' no existe")
            print("Ejecuta el paso 1 primero")
            return False
            
        extractor = LandmarkExtractor()
        summary = extractor.process_all_phrases(
            frames_base_dir=self.config['frames_dir'],
            output_base_dir=self.config['landmarks_dir']
        )
        
        if summary and summary['total_sequences'] > 0:
            print(f"✓ Landmarks extraídos exitosamente: {summary['total_sequences']} secuencias")
            return True
        else:
            print("✗ Error en la extracción de landmarks")
            return False
    
    def step3_create_dataset(self):
        """Paso 3: Crear dataset unificado"""
        print("\n=== PASO 3: CREACIÓN DE DATASET ===")
        
        extractor = LandmarkExtractor()
        dataset_file = os.path.join(self.config['dataset_dir'], 'lsa_sequence_dataset.pkl')
        
        dataset = extractor.create_unified_dataset(
            landmarks_base_dir=self.config['landmarks_dir'],
            output_file=dataset_file
        )
        
        if dataset and dataset['metadata']['total_sequences'] > 0:
            print(f"✓ Dataset creado exitosamente: {dataset_file}")
            print(f"  - Secuencias: {dataset['metadata']['total_sequences']}")
            print(f"  - Frases: {len(dataset['metadata']['phrases'])}")
            return True
        else:
            print("✗ Error en la creación del dataset")
            return False
    
    def step4_train_holistic_model(self):
        """Paso 4: Entrenar modelo holístico básico"""
        print("\n=== PASO 4: ENTRENAMIENTO DE MODELO HOLÍSTICO ===")
        
        if not os.path.exists(self.config['landmarks_dir']):
            print(f"Error: Directorio de landmarks '{self.config['landmarks_dir']}' no existe")
            print("Ejecuta el paso 2 primero")
            return False
        
        # Usar el nuevo script de entrenamiento holístico
        cmd = [sys.executable, 'train_holistic_model.py']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✓ Modelo holístico entrenado exitosamente")
                return True
            else:
                print("✗ Error en el entrenamiento del modelo")
                if result.stderr:
                    print("Error:", result.stderr)
                if result.stdout:
                    print("Output:", result.stdout)
                return False
                
        except Exception as e:
            print(f"✗ Error ejecutando entrenamiento: {e}")
            return False
    
    def step5_setup_gloss_converter(self):
        """Paso 5: Configurar convertidor de glosas"""
        print("\n=== PASO 5: CONFIGURACIÓN DE CONVERTIDOR DE GLOSAS ===")
        
        try:
            converter = GlossToTextConverter(
                method=self.config['gloss_method'],
                api_key=self.config['openai_api_key']
            )
            
            # Probar con ejemplos
            test_glosses = [
                "YO COMPRAR CARNE",
                "MAMÁ COCINAR COMIDA RICA",
                "MAÑANA TRABAJO IR"
            ]
            
            results = converter.convert_multiple_glosses(test_glosses)
            
            # Guardar resultados de prueba
            results_file = os.path.join(self.config['results_dir'], 'gloss_conversion_test.json')
            converter.save_results(results, results_file)
            
            print("✓ Convertidor de glosas configurado y probado exitosamente")
            print(f"  - Método: {self.config['gloss_method']}")
            print(f"  - Resultados de prueba: {results_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error configurando convertidor de glosas: {e}")
            return False
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo"""
        print("=== PIPELINE COMPLETO LSA → TEXTO ===")
        print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Extracción de frames", self.step1_extract_frames),
            ("Extracción de landmarks", self.step2_extract_landmarks),
            ("Entrenamiento de modelo holístico", self.step4_train_holistic_model),
            ("Configuración de convertidor", self.step5_setup_gloss_converter)
        ]
        
        results = {}
        
        for step_name, step_function in steps:
            print(f"\n{'='*60}")
            success = step_function()
            results[step_name] = success
            
            if not success:
                print(f"\n❌ Pipeline detenido en: {step_name}")
                break
        
        # Resumen final
        print(f"\n{'='*60}")
        print("=== RESUMEN DEL PIPELINE ===")
        
        for step_name, success in results.items():
            status = "✓ COMPLETADO" if success else "✗ FALLÓ"
            print(f"{step_name}: {status}")
        
        if all(results.values()):
            print(f"\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
            print(f"Tu traductor de LSA está listo para usar.")
            self._show_usage_instructions()
        else:
            print(f"\n⚠️  Pipeline incompleto. Revisa los errores arriba.")
        
        print(f"\nFinalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _show_usage_instructions(self):
        """Muestra instrucciones de uso"""
        print(f"\n=== INSTRUCCIONES DE USO ===")
        print(f"1. Para grabar nuevos videos:")
        print(f"   python video_collector.py")
        print(f"")
        print(f"2. Para procesar videos existentes:")
        print(f"   python lsa_pipeline.py --step frames")
        print(f"")
        print(f"3. Para usar el traductor completo:")
        print(f"   python app.py")
        print(f"")
        print(f"4. Para convertir glosas manualmente:")
        print(f"   python gloss_to_text.py")
    
    def run_single_step(self, step_name):
        """Ejecuta un solo paso del pipeline"""
        steps_map = {
            'frames': self.step1_extract_frames,
            'landmarks': self.step2_extract_landmarks,
            'dataset': self.step3_create_dataset,
            'train': self.step4_train_holistic_model,
            'gloss': self.step5_setup_gloss_converter
        }
        
        if step_name not in steps_map:
            print(f"Error: Paso '{step_name}' no válido")
            print(f"Pasos disponibles: {', '.join(steps_map.keys())}")
            return False
        
        print(f"Ejecutando paso: {step_name}")
        return steps_map[step_name]()
    
    def create_config_template(self, output_file='lsa_config.json'):
        """Crea un archivo de configuración de ejemplo"""
        config_template = {
            "videos_dir": "videos",
            "frames_dir": "processed_frames",
            "landmarks_dir": "landmarks_data",
            "dataset_dir": "dataset",
            "models_dir": "models",
            "results_dir": "results",
            "target_frames": 30,
            "model_type": "lstm",
            "gloss_method": "rules",
            "openai_api_key": null,
            "_comments": {
                "model_type": "Opciones: lstm, gru, transformer",
                "gloss_method": "Opciones: openai, transformers, rules",
                "openai_api_key": "Requerido solo si gloss_method es 'openai'"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_template, f, indent=4, ensure_ascii=False)
        
        print(f"Archivo de configuración creado: {output_file}")
        print("Edita este archivo para personalizar el pipeline")


def main():
    parser = argparse.ArgumentParser(description='Pipeline completo para traductor LSA')
    parser.add_argument('--config', type=str, help='Archivo de configuración')
    parser.add_argument('--step', type=str, 
                       choices=['frames', 'landmarks', 'dataset', 'train', 'gloss'],
                       help='Ejecutar solo un paso específico')
    parser.add_argument('--create-config', action='store_true',
                       help='Crear archivo de configuración de ejemplo')
    
    args = parser.parse_args()
    
    if args.create_config:
        pipeline = LSAPipeline()
        pipeline.create_config_template()
        return
    
    # Crear pipeline
    pipeline = LSAPipeline(config_file=args.config)
    
    if args.step:
        # Ejecutar paso específico
        success = pipeline.run_single_step(args.step)
        sys.exit(0 if success else 1)
    else:
        # Ejecutar pipeline completo
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
