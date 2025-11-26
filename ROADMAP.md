## Roadmap de aprendizaje para este proyecto (Intérprete de Lengua de Señas)

Basado en el repositorio, este proyecto usa: Python, OpenCV, MediaPipe Holistic, TensorFlow/Keras (LSTM bidireccional), scikit-learn, PyQt5, y una arquitectura de pipeline para visión por computadora en tiempo real.

A continuación, una ruta de aprendizaje práctica, con objetivos, conceptos clave, archivos del proyecto donde aplicarlos y entregables concretos.

### Semana 0 — Preparación
- **Objetivo**: Tener el entorno listo y entender el pipeline de alto nivel.
- **Aprende**: venv, `pip`, estructura de proyecto, lectura rápida del `README.md`.
- **Practica en repo**:
  - Ejecuta `setup_env.bat` (Windows) o `setup_env_gitbash.sh`.
  - Corre `python lstm_pipeline.py` y verifica la salida.
- **Entregable**: Pipeline corre de punta a punta con ejemplos mínimos.

### Semana 1 — Fundamentos de visión por computadora
- **Objetivo**: Capturar video y extraer frames de forma controlada.
- **Aprende**: OpenCV (captura, codecs, FPS), manejo de archivos, sampling temporal.
- **Archivos relevantes**: `video_collector.py`, `frame_extractor.py`.
- **Practica**:
  - Modifica FPS/intervalo de muestreo en `frame_extractor.py` y compara calidad/estabilidad.
  - Añade validaciones (si no hay cámara/video, mensajes claros).
- **Entregable**: Frames uniformes y consistentes para distintos videos.

### Semana 2 — Landmarks holísticos con MediaPipe
- **Objetivo**: Entender y extraer los 1662 puntos (pose + manos + cara).
- **Aprende**: MediaPipe Holistic, normalización, manejo de faltantes.
- **Archivos**: `landmark_extractor.py`, `sequence_data_processor.py`, `app_holistic.py`, `app_holistic_simple.py`.
- **Practica**:
  - Visualiza y depura landmarks en diferentes condiciones de luz.
  - Asegura padding/mascarado cuando falten frames o landmarks.
- **Entregable**: Secuencias de 30 frames × 1662 features listas en `models/sequence_dataset.pkl`.

### Semana 3 — Modelado secuencial con LSTM (Keras/TensorFlow)
- **Objetivo**: Diseñar, entrenar y evaluar un LSTM bidireccional para clasificación.
- **Aprende**: Keras Sequential, capas LSTM/Dropout, callbacks (EarlyStopping/ReduceLROnPlateau), overfitting/underfitting, métricas.
- **Archivos**: `train_lstm_model.py`, `model/lstm_sign_recognizer.py`, `model/sequence_model.py`.
- **Practica**:
  - Cambia profundidad/ancho del LSTM y observa val_accuracy/val_loss.
  - Implementa y compara regularización L2 y dropout.
- **Entregable**: `models/lsa_lstm_model.h5` con reporte de entrenamiento guardado.

### Semana 4 — Data engineering y augmentation temporal
- **Objetivo**: Robustecer el dataset pequeño con augmentations temporales.
- **Aprende**: ruido gaussiano, time-shift, speed perturbation, balanceo de clases.
- **Archivos**: `data_augmentation.py`, `sequence_data_processor.py`.
- **Practica**:
  - Ajusta `augmentation_factor`, `noise_factor`, `shift_range`, `speed_range`.
  - Evalúa impacto en estabilidad y generalización.
- **Entregable**: `models/sequence_dataset_augmented.pkl` y comparación de resultados vs original.

### Semana 5 — Inferencia en tiempo real y suavizado
- **Objetivo**: Pipeline en vivo con buffer de secuencias y salida estable.
- **Aprende**: buffers circulares, smoothing (promedios móviles/medianas), latencia vs estabilidad.
- **Archivos**: `app_holistic_simple.py`, `app_holistic.py`, `sequence_data_processor.py`.
- **Practica**:
  - Ajusta tamaño de buffer y ventana de suavizado; mide latencia percibida.
  - Añade barras/indicadores visuales de “buffer lleno”.
- **Entregable**: Demo en vivo con predicciones estables y feedback visual.

### Semana 6 — Interfaz y experiencia de usuario (PyQt5)
- **Objetivo**: Crear/ajustar UI para control y visualización.
- **Aprende**: señales/slots, timers, hilos ligeros para captura, render de video en widgets.
- **Archivos**: `app.py` (si integra UI), `app_holistic.py`.
- **Practica**:
  - Botones: iniciar/detener, selector de modelo, indicadores de estado.
  - No bloquear UI durante captura/inferencia (usar `QTimer`/threading seguro).
- **Entregable**: UI fluida que controla todo el flujo en tiempo real.

### Semana 7 — Evaluación, pruebas y depuración
- **Objetivo**: Medir, detectar y corregir errores sistemáticamente.
- **Aprende**: scikit-learn (matriz de confusión, clasificación), logging, pruebas funcionales.
- **Archivos**: `test_lstm_pipeline.py`, `test_holistic_pipeline.py`, `results/`.
- **Practica**:
  - Genera y guarda métricas/figuras en `results/`.
  - Pruebas para detectar datasets faltantes y rutas inválidas.
- **Entregable**: Reporte de métricas y scripts de verificación reproducibles.

### Semana 8 — Buenas prácticas, organización y despliegue local
- **Objetivo**: Dejar el proyecto sólido para mantenimiento/escala.
- **Aprende**: manejo de `.env` (si aplica), versionado de modelos/datasets, scripts CLI, empaquetado básico.
- **Archivos**: `check_dependencies.py`, `setup_env.*`, `README.md`, `ESTADO_PIPELINE.md`.
- **Practica**:
  - Comando único para “verificar requisitos” y “auto-fijar” problemas comunes.
  - Documentar parámetros y ejemplos reproducibles en el README.
- **Entregable**: Proyecto “listo para usar” por terceros en Windows.

---

## Conceptos clave a dominar (mapa mental)
- **Visión por computadora**: captura y preprocesado de video (OpenCV), sampling temporal.
- **Detección de landmarks**: MediaPipe Holistic, normalización y manejo de ausencias.
- **Deep Learning secuencial**: LSTM bidireccional, regularización, callbacks, tuning básico.
- **Ingeniería de datos temporal**: padding, máscaras, augmentations temporales.
- **Inferencia en tiempo real**: buffers, smoothing, trade-off latencia/estabilidad.
- **UX/GUI**: PyQt5 para control/visualización sin bloquear el hilo principal.
- **Evaluación**: métricas de clasificación y análisis de errores.
- **Mantenibilidad**: scripts de setup, verificación, documentación viva.

## Recorridos dentro del repo (para aprender haciendo)
- **Data → Secuencias**: `video_collector.py` → `frame_extractor.py` → `sequence_data_processor.py`
- **Entrenamiento**: `data_augmentation.py` → `train_lstm_model.py` → `model/lstm_sign_recognizer.py`
- **Inferencia/Aplicación**: `app_holistic_simple.py` / `app_holistic.py`
- **Automatización/Diagnóstico**: `lstm_pipeline.py`, `test_lstm_pipeline.py`, `check_dependencies.py`

## Siguientes pasos recomendados
1. Ejecuta y lee `lstm_pipeline.py` de punta a punta, siguiendo el `README.md`.
2. Abre y estudia `sequence_data_processor.py` y `model/lstm_sign_recognizer.py` para entender el formato exacto de entrada del modelo.
3. Practica ajustes controlados en `data_augmentation.py` y reentrena midiendo el impacto.
4. Mejora la estabilidad en `app_holistic_simple.py` ajustando el buffer y la estrategia de suavizado.

## Recursos sugeridos (complementarios)
- **OpenCV**: Documentación de `VideoCapture` y procesamiento básico.
- **MediaPipe Holistic**: Guía y ejemplos de Python.
- **Keras LSTM**: Guía de secuencias y RNNs en `keras.io`.
- **PyQt5**: Tutoriales de señales/slots y timers para video en tiempo real.


