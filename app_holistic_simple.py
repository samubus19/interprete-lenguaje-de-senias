import cv2
import numpy as np
from model.lstm_sign_recognizer import LSTMSignRecognizer
import os

# --- Constantes ---
STATUS_READY = "Listo para reconocer"
STATUS_RECOGNIZING = "Reconociendo..."
STATUS_NO_MODEL = "Modelo no cargado"

class SimpleLSTMApp:
    def __init__(self):
        """Aplicación simple para reconocimiento LSA usando solo OpenCV"""
        
        # Inicializar el reconocedor LSTM
        self.recognizer = LSTMSignRecognizer(model_path="models/lsa_lstm_model.h5")
        
        # Variables de estado
        self.is_recognizing = False
        self.current_prediction = "---"
        self.confidence = 0.0
        self.last_prediction = None
        self.prediction_count = 0
        self.min_predictions = 3  # Aumentado para evitar falsos positivos
        self.accumulated_text = ""
        
        # Filtros para evitar predicciones falsas
        self.hands_detected_count = 0
        self.min_hands_frames = 3  # Mínimo frames con manos para predecir
        self.no_hands_count = 0
        self.max_no_hands_tolerance = 15  # Tolerancia: frames sin manos antes de limpiar buffer (0.5s)
        self.buffer_cleared = False  # Flag para evitar limpiar repetidamente
        
        # Estado de la aplicación
        if self.recognizer.is_model_loaded():
            self.status = STATUS_READY
            print("✅ Modelo LSTM cargado correctamente")
            print(f"   Clases disponibles: {', '.join(self.recognizer.class_names)}")
        else:
            self.status = STATUS_NO_MODEL
            print("❌ Modelo LSTM no encontrado")
    
    def draw_interface(self, image):
        """Dibuja la interfaz simple en la imagen"""
        h, w = image.shape[:2]
        
        # Definir color de fondo según el estado
        if self.status == STATUS_RECOGNIZING:
            bg_color = (0, 200, 90)  # Verde
            text_color = (255, 255, 255)
        elif self.status == STATUS_NO_MODEL:
            bg_color = (0, 0, 200)   # Rojo
            text_color = (255, 255, 255)
        else:  # STATUS_READY
            bg_color = (245, 117, 16)  # Azul
            text_color = (255, 255, 255)
        
        # Fondo para la información
        cv2.rectangle(image, (0, 0), (w, 120), bg_color, -1)
        
        # Título
        cv2.putText(image, "LSA LSTM Reconocedor", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        # Estado actual
        cv2.putText(image, f"Estado: {self.status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Predicción actual
        cv2.putText(image, f"Detectado: {self.current_prediction}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # Confianza
        if self.confidence > 0:
            cv2.putText(image, f"Confianza: {self.confidence:.1%}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Buffer status si está reconociendo
        if self.is_recognizing and self.recognizer.is_model_loaded():
            buffer_status = self.recognizer.get_buffer_status()
            buffer_text = f"Buffer: {buffer_status['sequence_buffer_size']}/{buffer_status['sequence_buffer_max']}"
            cv2.putText(image, buffer_text, (w-200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Estado de detección de manos (solo si está reconociendo)
        if self.is_recognizing:
            hands_status = f"Manos: {self.hands_detected_count}/{self.min_hands_frames} | Sin manos: {self.no_hands_count}"
            cv2.putText(image, hands_status, (w-300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        
        # Instrucciones
        instructions = [
            "ESPACIO: Iniciar/Detener reconocimiento",
            "C: Limpiar texto acumulado", 
            "Q: Salir"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (10, h - 60 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Texto acumulado (si hay)
        if self.accumulated_text:
            # Fondo para texto acumulado
            cv2.rectangle(image, (0, 130), (w, 180), (50, 50, 50), -1)
            cv2.putText(image, "Frases reconocidas:", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Dividir texto largo en múltiples líneas
            max_chars = 80
            if len(self.accumulated_text) > max_chars:
                text_to_show = "..." + self.accumulated_text[-max_chars:]
            else:
                text_to_show = self.accumulated_text
                
            cv2.putText(image, text_to_show, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def has_valid_hands(self, results):
        """Verifica si hay manos válidas detectadas"""
        if not results:
            return False
        
        # Verificar si hay landmarks de manos
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None
        
        return has_left or has_right
    
    def process_recognition(self, frame):
        """Procesa el reconocimiento de señas con filtros anti-overfitting"""
        if not self.recognizer.is_model_loaded():
            return
        
        # Actualizar buffer de secuencia
        has_detection, results = self.recognizer.update_sequence_buffer_optimized(frame)
        
        # Verificar si hay manos válidas
        hands_present = self.has_valid_hands(results)
        
        if hands_present:
            self.hands_detected_count += 1
            self.no_hands_count = 0
            self.buffer_cleared = False  # Reset flag cuando vuelven las manos
        else:
            self.no_hands_count += 1
            # Solo limpiar buffer si hay ausencia prolongada de manos (no pérdidas momentáneas)
            if self.no_hands_count > self.max_no_hands_tolerance and not self.buffer_cleared:
                # Ausencia prolongada = usuario bajó las manos o salió de cámara
                print(f"🧹 Limpiando buffer tras {self.no_hands_count} frames sin manos (ausencia prolongada)")
                self.hands_detected_count = 0
                self.prediction_count = 0
                self.last_prediction = None
                self.recognizer.reset_buffers()  # Limpiar buffer del reconocedor
                self.buffer_cleared = True  # Evitar limpiar repetidamente
        
        # Obtener estado del buffer
        buffer_status = self.recognizer.get_buffer_status()
        
        # Hacer predicción SOLO si:
        # 1. El buffer está lleno
        # 2. Se han detectado manos suficientes veces
        # 3. Actualmente hay manos presentes
        prediction = None
        if (buffer_status['is_ready'] and 
            self.hands_detected_count >= self.min_hands_frames and 
            hands_present):
            prediction = self.recognizer.predict_sequence(threshold=0.80)  # Umbral al 80%
        
        if prediction:
            sign = prediction['sign']
            confidence = prediction['confidence']
            
            # Actualizar predicción actual
            self.current_prediction = sign.replace('_', ' ')
            self.confidence = confidence
            
            # Lógica para agregar señas al texto acumulado
            if sign != self.last_prediction:
                self.prediction_count = 1
                self.last_prediction = sign
            else:
                self.prediction_count += 1
            
            # Si la predicción es consistente, agregarla al texto
            if self.prediction_count >= self.min_predictions:
                # Evitar duplicados consecutivos en el texto acumulado
                new_sign = sign.replace('_', ' ')
                if not self.accumulated_text.endswith(new_sign):
                    if self.accumulated_text:
                        self.accumulated_text += " | "
                    self.accumulated_text += new_sign
                    print(f"✅ Seña agregada: {new_sign} (conf: {confidence:.1%})")
                
                # Reset para permitir nueva detección de la misma seña después de pausa
                self.prediction_count = 0
                self.last_prediction = None  # Permite re-detectar la misma seña
        else:
            if not hands_present:
                if self.no_hands_count > self.max_no_hands_tolerance:
                    self.current_prediction = "Esperando manos (buffer limpio)"
                else:
                    self.current_prediction = f"Sin manos ({self.no_hands_count}/{self.max_no_hands_tolerance})"
            elif self.hands_detected_count < self.min_hands_frames:
                self.current_prediction = f"Detectando manos... ({self.hands_detected_count}/{self.min_hands_frames})"
            elif buffer_status['is_ready']:
                self.current_prediction = "Confianza muy baja"
            else:
                self.current_prediction = f"Llenando buffer... ({buffer_status['sequence_buffer_size']}/30)"
            self.confidence = 0.0
        
        return results
    
    def draw_hands_only(self, image, results):
        """Dibuja solo las manos para mejor rendimiento"""
        if not results:
            return
        
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        
        # Dibujar mano izquierda
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
            )
        
        # Dibujar mano derecha
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
            )
    
    def run(self):
        """Ejecuta la aplicación principal"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara.")
            return
        
        # Configurar cámara para mejor rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # FPS alto para captura, procesamos menos
        
        print("\n🚀 Aplicación iniciada")
        print("📋 Controles:")
        print("   ESPACIO: Iniciar/Detener reconocimiento")
        print("   C: Limpiar texto acumulado")
        print("   Q: Salir")
        
        frame_count = 0
        process_every_n_frames = 1  # Procesar cada 1 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: No se pudo leer el frame.")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar reconocimiento solo cada N frames y si está activo
            results = None
            if self.is_recognizing and frame_count % process_every_n_frames == 0:
                if self.recognizer.is_model_loaded():
                    results = self.process_recognition(frame)
            
            # Dibujar manos si hay resultados
            if results and self.is_recognizing:
                self.draw_hands_only(frame, results)
            
            # Dibujar interfaz
            self.draw_interface(frame)
            
            # Mostrar frame
            cv2.imshow('LSA LSTM Reconocedor - Simple', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # ESPACIO
                if self.recognizer.is_model_loaded():
                    self.is_recognizing = not self.is_recognizing
                    if self.is_recognizing:
                        self.status = STATUS_RECOGNIZING
                        self.recognizer.reset_buffers()
                        print("🔴 Reconocimiento INICIADO")
                    else:
                        self.status = STATUS_READY
                        self.current_prediction = "---"
                        self.confidence = 0.0
                        print("⏸️ Reconocimiento DETENIDO")
                else:
                    print("❌ No se puede iniciar - Modelo no cargado")
            elif key == ord('c'):  # Limpiar
                self.accumulated_text = ""
                self.current_prediction = "---"
                self.confidence = 0.0
                self.last_prediction = None
                self.prediction_count = 0
                self.hands_detected_count = 0
                self.no_hands_count = 0
                self.buffer_cleared = False  # Reset flag de limpieza
                if self.recognizer.is_model_loaded():
                    self.recognizer.reset_buffers()
                print("🗑️ Texto y contadores limpiados")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Aplicación cerrada")

def main():
    # Verificar si el modelo existe
    if not os.path.exists("models/lsa_lstm_model.h5"):
        print("❌ Modelo LSTM no encontrado!")
        print("   Ejecuta: python sequence_data_processor.py")
        print("   Luego: python train_lstm_model.py")
        return
    
    app = SimpleLSTMApp()
    app.run()

if __name__ == '__main__':
    main()
