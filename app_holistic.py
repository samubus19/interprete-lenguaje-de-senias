import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from model.lstm_sign_recognizer import LSTMSignRecognizer

class LSTMSignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intérprete LSA - Modelo LSTM")
        self.setGeometry(100, 100, 1400, 900)

        # Inicializar el reconocedor LSTM
        self.recognizer = LSTMSignRecognizer(model_path="models/lsa_lstm_model.h5")
        
        # Variables de estado
        self.current_phrase = ""
        self.last_prediction = None
        self.prediction_count = 0
        self.min_predictions = 2  # Mínimo de predicciones consistentes para LSTM

        # Configuración de MediaPipe para dibujo
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Configuración de la cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Timer para actualizar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(150)  # ~7 FPS para LSTM (más lento pero más preciso)

        # Configurar interfaz
        self.setup_ui()
        
        # Estado de procesamiento
        self.is_processing = False
        
        # Verificar si el modelo está cargado
        if not self.recognizer.is_model_loaded():
            self.status_label.setText("❌ MODELO NO CARGADO - Entrena el modelo primero")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setText("✅ Modelo LSTM listo")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(self.central_widget)
        
        # Panel izquierdo - Video
        left_panel = QVBoxLayout()
        
        # Etiqueta para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        left_panel.addWidget(self.video_label)
        
        # Controles
        controls_layout = QHBoxLayout()
        
        self.toggle_button = QPushButton("▶️ Iniciar Reconocimiento")
        self.toggle_button.clicked.connect(self.toggle_recognition)
        self.toggle_button.setMinimumHeight(50)
        controls_layout.addWidget(self.toggle_button)
        
        self.clear_button = QPushButton("🗑️ Limpiar")
        self.clear_button.clicked.connect(self.clear_text)
        self.clear_button.setMinimumHeight(50)
        controls_layout.addWidget(self.clear_button)
        
        left_panel.addLayout(controls_layout)
        
        # Panel derecho - Resultados
        right_panel = QVBoxLayout()
        
        # Título
        title_label = QLabel("🤟 Traductor LSA LSTM")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        right_panel.addWidget(title_label)
        
        # Estado del modelo
        self.status_label = QLabel("Cargando modelo...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        right_panel.addWidget(self.status_label)
        
        # Predicción actual
        current_label = QLabel("Predicción Actual:")
        current_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_panel.addWidget(current_label)
        
        self.current_prediction_label = QLabel("---")
        self.current_prediction_label.setAlignment(Qt.AlignCenter)
        self.current_prediction_label.setFont(QFont("Arial", 14))
        self.current_prediction_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.current_prediction_label.setMinimumHeight(60)
        right_panel.addWidget(self.current_prediction_label)
        
        # Confianza
        self.confidence_label = QLabel("Confianza: ---%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 10))
        right_panel.addWidget(self.confidence_label)
        
        # Barra de progreso del buffer
        buffer_label = QLabel("Buffer de Secuencia:")
        buffer_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_panel.addWidget(buffer_label)
        
        self.buffer_progress = QProgressBar()
        self.buffer_progress.setMaximum(30)  # max_sequence_length
        self.buffer_progress.setTextVisible(True)
        self.buffer_progress.setFormat("%v/%m frames")
        right_panel.addWidget(self.buffer_progress)
        
        # Texto acumulado
        accumulated_label = QLabel("Frases Reconocidas:")
        accumulated_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_panel.addWidget(accumulated_label)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Arial", 11))
        self.text_area.setMinimumHeight(200)
        right_panel.addWidget(self.text_area)
        
        # Información del modelo
        info_label = QLabel("Información del Modelo:")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_panel.addWidget(info_label)
        
        self.model_info_label = QLabel(self.recognizer.get_model_info())
        self.model_info_label.setFont(QFont("Arial", 9))
        self.model_info_label.setStyleSheet("background-color: #f8f8f8; padding: 5px; border-radius: 3px;")
        self.model_info_label.setWordWrap(True)
        right_panel.addWidget(self.model_info_label)
        
        # Agregar paneles al layout principal
        main_layout.addLayout(left_panel, 2)  # 2/3 del espacio
        main_layout.addLayout(right_panel, 1)  # 1/3 del espacio

    def update_frame(self):
        """Actualiza el frame de video y procesa reconocimiento"""
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        if self.is_processing and self.recognizer.is_model_loaded():
            # Actualizar buffer de secuencia
            has_detection = self.recognizer.update_sequence_buffer(frame)
            
            # Actualizar barra de progreso
            buffer_status = self.recognizer.get_buffer_status()
            self.buffer_progress.setValue(buffer_status['sequence_buffer_size'])
            
            # Hacer predicción solo si el buffer está lleno
            prediction = None
            if buffer_status['is_ready']:
                prediction = self.recognizer.predict_sequence(threshold=0.6)
            
            if prediction:
                sign = prediction['sign']
                confidence = prediction['confidence']
                
                # Actualizar predicción actual
                self.current_prediction_label.setText(sign.replace('_', ' '))
                self.confidence_label.setText(f"Confianza: {confidence:.1%} ({prediction.get('buffer_count', 0)}/{prediction.get('buffer_size', 0)})")
                
                # Lógica para agregar señas al texto acumulado
                if sign != self.last_prediction:
                    self.prediction_count = 1
                    self.last_prediction = sign
                else:
                    self.prediction_count += 1
                
                # Si la predicción es consistente, agregarla al texto
                if self.prediction_count >= self.min_predictions:
                    if sign not in self.current_phrase:
                        if self.current_phrase:
                            self.current_phrase += " | "
                        self.current_phrase += sign.replace('_', ' ')
                        self.text_area.setText(self.current_phrase)
                        self.prediction_count = 0  # Reset counter
            else:
                if buffer_status['is_ready']:
                    self.current_prediction_label.setText("Sin detección")
                else:
                    self.current_prediction_label.setText(f"Llenando buffer... ({buffer_status['sequence_buffer_size']}/{buffer_status['sequence_buffer_max']})")
                self.confidence_label.setText("Confianza: ---%")
            
            # Dibujar landmarks en el frame
            frame = self.draw_landmarks(frame)
        
        # Convertir frame para mostrar en Qt
        self.display_frame(frame)

    def draw_landmarks(self, frame):
        """Dibuja landmarks en el frame"""
        try:
            results = self.recognizer.get_landmarks_for_drawing(frame)
            
            if results:
                # Dibujar pose
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, 
                        self.mp_holistic.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                
                # Dibujar manos
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, 
                        self.mp_holistic.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, 
                        self.mp_holistic.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Dibujar cara (solo contorno)
                if results.face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.face_landmarks, 
                        self.mp_holistic.FACEMESH_CONTOURS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
                    )
        except Exception as e:
            print(f"Error dibujando landmarks: {e}")
        
        return frame

    def display_frame(self, frame):
        """Convierte y muestra el frame en la interfaz"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Escalar manteniendo proporción
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def toggle_recognition(self):
        """Inicia/detiene el reconocimiento"""
        if not self.recognizer.is_model_loaded():
            self.status_label.setText("❌ No se puede iniciar - Modelo no cargado")
            return
            
        self.is_processing = not self.is_processing
        
        if self.is_processing:
            self.toggle_button.setText("⏸️ Detener Reconocimiento")
            self.toggle_button.setStyleSheet("background-color: #ff6b6b;")
            self.status_label.setText("🔴 RECONOCIENDO... (LSTM)")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.recognizer.reset_buffers()
        else:
            self.toggle_button.setText("▶️ Iniciar Reconocimiento")
            self.toggle_button.setStyleSheet("")
            self.status_label.setText("✅ Modelo LSTM listo")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.current_prediction_label.setText("---")
            self.confidence_label.setText("Confianza: ---%")

    def clear_text(self):
        """Limpia el texto acumulado"""
        self.current_phrase = ""
        self.text_area.clear()
        self.current_prediction_label.setText("---")
        self.confidence_label.setText("Confianza: ---%")
        self.last_prediction = None
        self.prediction_count = 0
        self.recognizer.reset_buffers()
        self.buffer_progress.setValue(0)

    def closeEvent(self, event):
        """Limpia recursos al cerrar"""
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Verificar si el modelo existe
    import os
    if not os.path.exists("models/lsa_lstm_model.h5"):
        print("❌ Modelo LSTM no encontrado!")
        print("   Ejecuta: python sequence_data_processor.py")
        print("   Luego: python train_lstm_model.py")
        return
    
    window = LSTMSignLanguageApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
