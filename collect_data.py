import os
import json
import cv2
import numpy as np
from datetime import datetime
from model.data_collector import DataCollector

def main():
    # Crear directorio para datos si no existe
    os.makedirs('data/collected', exist_ok=True)
    
    # Cargar señas disponibles
    with open("data/signs.json", "r", encoding="utf-8") as f:
        signs_data = json.load(f)
    
    # Inicializar recolector de datos
    collector = DataCollector()
    
    print("\n=== Recolección de Datos de Lenguaje de Señas ===")
    print("\nSeñas disponibles:")
    for sign in signs_data["signs"]:
        print(f"{sign['id']}: {sign['name']} - {sign['description']}")
    
    # Seleccionar señas a grabar
    print("\nIngresa los IDs de las señas que quieres grabar (separados por coma)")
    print("Por ejemplo: 1,3,5")
    print("O presiona Enter para grabar todas las señas")
    
    selected_ids = input("\nIDs de señas a grabar: ").strip()
    
    if selected_ids:
        try:
            selected_ids = [int(id.strip()) for id in selected_ids.split(",")]
            signs_to_record = [sign for sign in signs_data["signs"] if sign["id"] in selected_ids]
        except ValueError:
            print("Error: Ingresa números válidos separados por coma")
            return
    else:
        signs_to_record = signs_data["signs"]
    
    print("\nInstrucciones:")
    print("1. Para cada seña, muestra la seña frente a la cámara")
    print("2. Presiona 'c' para capturar una muestra")
    print("3. Presiona 'q' para pasar a la siguiente seña")
    print("4. Presiona 'ESC' para terminar la recolección\n")
    
    # Recolectar datos para cada seña seleccionada
    for sign in signs_to_record:
        sign_id = sign["id"]
        sign_name = sign["name"]
        
        print(f"\n=== Recolectando datos para la seña: {sign_name} (ID: {sign_id}) ===")
        print("Presiona 'c' para capturar, 'q' para siguiente seña, 'ESC' para terminar")
        
        # Crear directorio para la seña si no existe
        sign_dir = os.path.join('data/collected', str(sign_id))
        os.makedirs(sign_dir, exist_ok=True)
        
        samples = 0
        while samples < 50:  # Recolectar 50 muestras por seña
            frame = collector.capture_frame()
            if frame is None:
                continue
                
            # Mostrar frame
            cv2.imshow('Recolección de Datos', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\nRecolección terminada por el usuario.")
                return
            elif key == ord('q'):
                print(f"\nPasando a la siguiente seña. Muestras recolectadas: {samples}")
                break
            elif key == ord('c'):
                landmarks = collector.detect_landmarks(frame)
                if landmarks:
                    processed_data = collector.preprocess_landmarks(landmarks)
                    if processed_data is not None:
                        # Guardar datos
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"sample_{timestamp}.npy"
                        filepath = os.path.join(sign_dir, filename)
                        np.save(filepath, processed_data)
                        
                        samples += 1
                        print(f"Muestra {samples}/50 capturada")
    
    print("\nRecolección de datos completada!")
    print("Datos guardados en el directorio 'data/collected'")

if __name__ == "__main__":
    main() 