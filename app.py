import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from model.sign_recognizer import SignRecognizer

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intérprete de Lenguaje de Señas Argentino")
        self.setGeometry(100, 100, 1200, 800)

        # Inicializar el reconocedor de señas
        self.recognizer = SignRecognizer()

        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Configuración de la cámara
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Configuración de la interfaz
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Etiqueta para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Área de texto para mostrar el texto interpretado
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setMaximumHeight(100)
        self.layout.addWidget(self.text_area)

        # Botón para iniciar/detener la interpretación
        self.toggle_button = QPushButton("Iniciar Interpretación")
        self.toggle_button.clicked.connect(self.toggle_interpretation)
        self.layout.addWidget(self.toggle_button)

        self.is_processing = False
        self.last_predictions = []

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir el frame a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.is_processing:
                # Procesar el frame con el reconocedor de señas
                predictions = self.recognizer.predict(frame)
                
                if predictions:
                    self.last_predictions = predictions
                    text = "Señas detectadas:\n"
                    for pred in predictions:
                        text += f"- {pred['sign']} (confianza: {pred['confidence']:.2f})\n"
                        # Dibuja el nombre y confianza sobre el video
                        cv2.putText(
                            rgb_frame,
                            f"{pred['sign']} ({pred['confidence']*100:.1f}%)",
                            (30, 60),  # Puedes ajustar la posición
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3
                        )
                    self.text_area.setText(text)
                else:
                    self.text_area.setText("Sin seña detectada")
                
                # Dibujar los landmarks de las manos
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

            # Convertir el frame a QImage para mostrarlo
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))

    def toggle_interpretation(self):
        self.is_processing = not self.is_processing
        self.toggle_button.setText(
            "Detener Interpretación" if self.is_processing else "Iniciar Interpretación"
        )
        if not self.is_processing:
            self.text_area.clear()
            self.last_predictions = []

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_()) 