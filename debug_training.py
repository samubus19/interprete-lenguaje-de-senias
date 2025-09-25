#!/usr/bin/env python3
"""
Script de debug para el entrenamiento del modelo
"""

import os
import pickle
import numpy as np
from model.sequence_model import LSASequenceModel

def debug_dataset():
    """Debug del dataset"""
    dataset_path = 'dataset/lsa_sequence_dataset.pkl'
    
    print("=== DEBUG DATASET ===")
    print(f"Verificando archivo: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset no existe")
        return False
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print("✅ Dataset cargado exitosamente")
        print(f"Claves del dataset: {list(dataset.keys())}")
        
        if 'sequences' in dataset:
            sequences = dataset['sequences']
            print(f"Número de secuencias: {len(sequences)}")
            
            if len(sequences) > 0:
                print(f"Forma de la primera secuencia: {sequences[0].shape}")
                print(f"Tipo de datos: {type(sequences[0])}")
        
        if 'labels' in dataset:
            labels = dataset['labels']
            print(f"Número de etiquetas: {len(labels)}")
            print(f"Etiquetas únicas: {set(labels)}")
        
        if 'metadata' in dataset:
            metadata = dataset['metadata']
            print(f"Metadatos: {metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_model_creation():
    """Debug de la creación del modelo"""
    print("\n=== DEBUG CREACIÓN DEL MODELO ===")
    
    try:
        # Crear modelo con parámetros básicos
        model = LSASequenceModel(max_sequence_length=30, feature_dim=378, num_classes=5)
        print("✅ Modelo LSASequenceModel creado")
        
        # Intentar construir el modelo
        model.build_model(model_type='lstm')
        print("✅ Modelo LSTM construido")
        
        # Mostrar resumen
        print("Resumen del modelo:")
        model.model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_data_preparation():
    """Debug de la preparación de datos"""
    print("\n=== DEBUG PREPARACIÓN DE DATOS ===")
    
    dataset_path = 'dataset/lsa_sequence_dataset.pkl'
    
    try:
        model = LSASequenceModel(max_sequence_length=30)
        X_train, X_val, y_train, y_val, class_names = model.prepare_data(dataset_path)
        
        print("✅ Datos preparados exitosamente")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"Nombres de clases: {class_names}")
        print(f"Número de clases: {len(class_names)}")
        
        return True, model, X_train, X_val, y_train, y_val, class_names
        
    except Exception as e:
        print(f"❌ Error preparando datos: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None, None, None

def debug_training():
    """Debug del entrenamiento"""
    print("\n=== DEBUG ENTRENAMIENTO ===")
    
    success, model, X_train, X_val, y_train, y_val, class_names = debug_data_preparation()
    
    if not success:
        return False
    
    try:
        # Construir modelo
        model.build_model(model_type='lstm')
        print("✅ Modelo construido para entrenamiento")
        
        # Entrenar por pocas épocas para probar
        print("Iniciando entrenamiento de prueba (2 épocas)...")
        history = model.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=4)
        
        print("✅ Entrenamiento completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== DEBUG COMPLETO DEL ENTRENAMIENTO ===")
    
    # 1. Debug del dataset
    if not debug_dataset():
        print("❌ Fallo en debug del dataset")
        exit(1)
    
    # 2. Debug de creación del modelo
    if not debug_model_creation():
        print("❌ Fallo en debug de creación del modelo")
        exit(1)
    
    # 3. Debug del entrenamiento
    if not debug_training():
        print("❌ Fallo en debug del entrenamiento")
        exit(1)
    
    print("\n✅ Todos los tests de debug pasaron exitosamente!")
    print("El entrenamiento debería funcionar correctamente.")
