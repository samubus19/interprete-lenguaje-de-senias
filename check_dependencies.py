#!/usr/bin/env python3
"""
Script para verificar dependencias del proyecto LSA
"""

import sys
import subprocess

def check_dependency(package_name, import_name=None):
    """Verifica si una dependencia está instalada"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} - INSTALADO")
        return True
    except ImportError:
        print(f"✗ {package_name} - FALTANTE")
        return False

def install_missing_dependencies():
    """Instala dependencias faltantes"""
    missing_packages = []
    
    # Lista de dependencias críticas
    dependencies = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('mediapipe', 'mediapipe'),
        ('scikit-learn', 'sklearn'),
        ('tensorflow', 'tensorflow'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pathlib', 'pathlib'),
        ('pickle', 'pickle'),
        ('json', 'json'),
        ('os', 'os'),
        ('sys', 'sys'),
        ('argparse', 'argparse'),
        ('subprocess', 'subprocess'),
        ('datetime', 'datetime')
    ]
    
    print("=== VERIFICACIÓN DE DEPENDENCIAS ===")
    
    for package, import_name in dependencies:
        if not check_dependency(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n=== DEPENDENCIAS FALTANTES ===")
        for package in missing_packages:
            print(f"- {package}")
        
        print(f"\n=== INSTALANDO DEPENDENCIAS FALTANTES ===")
        for package in missing_packages:
            if package not in ['pathlib', 'pickle', 'json', 'os', 'sys', 'argparse', 'subprocess', 'datetime']:
                print(f"Instalando {package}...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✓ {package} instalado exitosamente")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Error instalando {package}: {e}")
    else:
        print("\n✓ Todas las dependencias están instaladas")

def test_imports():
    """Prueba las importaciones específicas del proyecto"""
    print("\n=== PROBANDO IMPORTACIONES DEL PROYECTO ===")
    
    try:
        import cv2
        print("✓ OpenCV importado correctamente")
    except Exception as e:
        print(f"✗ Error importando OpenCV: {e}")
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe importado correctamente")
    except Exception as e:
        print(f"✗ Error importando MediaPipe: {e}")
    
    try:
        from sklearn.cluster import KMeans
        print("✓ scikit-learn importado correctamente")
    except Exception as e:
        print(f"✗ Error importando scikit-learn: {e}")
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow importado correctamente")
        print(f"  Versión: {tf.__version__}")
    except Exception as e:
        print(f"✗ Error importando TensorFlow: {e}")
    
    try:
        import numpy as np
        print("✓ NumPy importado correctamente")
        print(f"  Versión: {np.__version__}")
    except Exception as e:
        print(f"✗ Error importando NumPy: {e}")

def test_project_imports():
    """Prueba las importaciones específicas del proyecto"""
    print("\n=== PROBANDO IMPORTACIONES DEL PROYECTO LSA ===")
    
    try:
        from frame_extractor import FrameExtractor
        print("✓ FrameExtractor importado correctamente")
    except Exception as e:
        print(f"✗ Error importando FrameExtractor: {e}")
    
    try:
        from landmark_extractor import LandmarkExtractor
        print("✓ LandmarkExtractor importado correctamente")
    except Exception as e:
        print(f"✗ Error importando LandmarkExtractor: {e}")
    
    try:
        from model.sequence_model import LSASequenceModel
        print("✓ LSASequenceModel importado correctamente")
    except Exception as e:
        print(f"✗ Error importando LSASequenceModel: {e}")
    
    try:
        from gloss_to_text import GlossToTextConverter
        print("✓ GlossToTextConverter importado correctamente")
    except Exception as e:
        print(f"✗ Error importando GlossToTextConverter: {e}")

if __name__ == "__main__":
    print("=== DIAGNÓSTICO DE DEPENDENCIAS LSA ===")
    print(f"Python: {sys.version}")
    print(f"Directorio actual: {sys.path[0]}")
    
    install_missing_dependencies()
    test_imports()
    test_project_imports()
    
    print("\n=== DIAGNÓSTICO COMPLETADO ===")
    print("Si hay errores, instala las dependencias faltantes con:")
    print("pip install opencv-python mediapipe scikit-learn tensorflow matplotlib seaborn")
