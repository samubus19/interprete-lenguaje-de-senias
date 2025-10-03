#!/usr/bin/env python3
"""
Script para probar el modelo holístico entrenado
"""

import os
import cv2
import numpy as np
from model.holistic_sign_recognizer import HolisticSignRecognizer

def test_model_loading():
    """Prueba la carga del modelo"""
    print("=== PRUEBA DE CARGA DEL MODELO ===")
    
    recognizer = HolisticSignRecognizer()
    
    if recognizer.is_model_loaded():
        print("✅ Modelo cargado correctamente")
        print(recognizer.get_model_info())
        return True
    else:
        print("❌ Error cargando el modelo")
        print("   Ejecuta: python train_holistic_model.py")
        return False

def test_webcam_recognition():
    """Prueba el reconocimiento en tiempo real con webcam"""
    print("\n=== PRUEBA DE RECONOCIMIENTO EN TIEMPO REAL ===")
    
    recognizer = HolisticSignRecognizer()
    
    if not recognizer.is_model_loaded():
        print("❌ Modelo no cargado")
        return False
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se puede acceder a la cámara")
        return False
    
    print("✅ Cámara inicializada")
    print("📹 Presiona 'q' para salir, 'r' para resetear buffer")
    print("🤟 Haz señas frente a la cámara...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Voltear para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Hacer predicción cada 5 frames (para mejor rendimiento)
            if frame_count % 5 == 0:
                prediction = recognizer.predict(frame, threshold=0.4)
                
                if prediction:
                    phrase = prediction['phrase']
                    confidence = prediction['confidence']
                    
                    # Mostrar predicción en el frame
                    text = f"{phrase.replace('_', ' ')} ({confidence:.1%})"
                    cv2.putText(frame, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"🎯 {phrase.replace('_', ' ')} - Confianza: {confidence:.1%}")
                else:
                    cv2.putText(frame, "Sin deteccion", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Dibujar landmarks
            try:
                results = recognizer.get_landmarks_for_drawing(frame)
                if results and results.pose_landmarks:
                    # Dibujar solo las manos para mejor visualización
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_holistic = mp.solutions.holistic
                    
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.left_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS)
                    
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.right_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS)
            except:
                pass  # Ignorar errores de dibujo
            
            # Mostrar frame
            cv2.imshow('Prueba Modelo Holístico LSA', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.reset_buffer()
                print("🔄 Buffer reseteado")
    
    except KeyboardInterrupt:
        print("\n⏹️ Detenido por el usuario")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados")
    
    return True

def test_static_image():
    """Prueba con una imagen estática si existe"""
    print("\n=== PRUEBA CON IMAGEN ESTÁTICA ===")
    
    # Buscar imágenes de prueba
    test_dirs = ['processed_frames', 'test_images', '.']
    test_image = None
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(test_dir, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("⚠️ No se encontraron imágenes de prueba")
        return False
    
    print(f"📸 Probando con imagen: {test_image}")
    
    recognizer = HolisticSignRecognizer()
    if not recognizer.is_model_loaded():
        return False
    
    # Cargar y procesar imagen
    image = cv2.imread(test_image)
    if image is None:
        print("❌ Error cargando imagen")
        return False
    
    prediction = recognizer.predict(image)
    
    if prediction:
        print(f"✅ Predicción: {prediction['phrase'].replace('_', ' ')}")
        print(f"   Confianza: {prediction['confidence']:.1%}")
    else:
        print("❌ No se pudo hacer predicción")
    
    return True

def main():
    """Función principal de pruebas"""
    print("🤟 PRUEBAS DEL MODELO HOLÍSTICO LSA")
    print("=" * 50)
    
    # Verificar archivos necesarios
    required_files = [
        "models/holistic_sign_model.h5",
        "models/holistic_model_metadata.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Ejecuta primero:")
        print("   python train_holistic_model.py")
        print("   O: python lsa_pipeline.py --step train")
        return
    
    # Ejecutar pruebas
    tests = [
        ("Carga del modelo", test_model_loading),
        ("Imagen estática", test_static_image),
        ("Reconocimiento en tiempo real", test_webcam_recognition)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print("🎯 Para usar la aplicación completa:")
    print("   python app_holistic.py")

if __name__ == "__main__":
    main()
