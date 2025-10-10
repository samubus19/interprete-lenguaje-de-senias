#  Traductor de Lengua de Señas Argentina (LSA) a Texto

Sistema completo de traducción de LSA a texto natural utilizando **modelo LSTM con landmarks holísticos**.

##  Características principales

- **Modelo LSTM avanzado** para reconocimiento de secuencias LSA
- **Landmarks holísticos** con MediaPipe (1662 características: pose + manos + cara)
- **Procesamiento de secuencias** de 30 frames por video
- **Data augmentation** automático para evitar overfitting
- **Aplicación en tiempo real** con buffer de secuencias
- **Pipeline completo automatizado** desde grabación hasta predicción
- **Prevención de overfitting** con técnicas avanzadas

## Requisitos

- Python 3.8 o superior
- Cámara web
- 4GB RAM mínimo (8GB recomendado)
- Espacio en disco: 2GB para modelos y datos

### Dependencias principales
- TensorFlow 2.x
- MediaPipe
- OpenCV
- scikit-learn
- transformers (opcional)
- openai (opcional)

## Instalación Rápida

### 1. Configurar Entorno Virtual

#### Windows (Git Bash):
```bash
chmod +x setup_env_gitbash.sh
./setup_env_gitbash.sh
```

#### Windows (Command Prompt):
```cmd
setup_env.bat
```

#### Linux/Mac:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Instalación Manual (Alternativa)
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# venv\Scripts\activate       # Windows CMD
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

## 🚀 Guía de Uso Completa

### ⚡ Inicio Rápido (Para Principiantes)

Si es tu primera vez usando el sistema, tienes **2 opciones**:

#### **Opción 1: Pipeline Automatizado (Recomendado) 🤖**
```bash
# Ejecuta todo el pipeline completo automáticamente
python lstm_pipeline.py
```

#### **Opción 2: Paso a Paso Manual 🔧**
```bash
# 1. Grabar videos de señas
python video_collector.py

# 2. Extraer frames de los videos (30 por video)
python frame_extractor.py

# 3. Procesar frames a secuencias de landmarks
python sequence_data_processor.py

# 4. Data augmentation (aumenta datos 20x)
python data_augmentation.py

# 5. Entrenar modelo LSTM
python train_lstm_model.py

# 6. Usar la aplicación en tiempo real
python app_holistic_simple.py
```

> **💡 Recomendación**: Usa la **Opción 1** si es tu primera vez. El pipeline automatizado maneja errores y te guía paso a paso.

### 📋 Pipeline Completo Paso a Paso

#### **Paso 1: Grabación de Videos** 📹
```bash
python video_collector.py
```
**¿Qué hace?**
- Graba videos automáticamente al detectar manos
- Estructura: `./videos/FRASE/1.avi`
- **Recomendación**: Graba al menos 50-100 videos por seña

#### **Paso 2: Extracción de Frames** 🎬
```bash
python frame_extractor.py
```
**¿Qué hace?**
- Lee videos de `videos/FRASE/X.avi`
- Extrae 30 frames uniformes por video
- Guarda frames en `processed_frames/FRASE/video_X/frame_001.jpg`

#### **Paso 3: Procesamiento de Secuencias** 🔄
```bash
python sequence_data_processor.py
```
**¿Qué hace?**
- Lee frames de `processed_frames/`
- Extrae 1662 landmarks holísticos por frame (pose + manos + cara)
- Crea dataset de secuencias: `models/sequence_dataset.pkl`

#### **Paso 4: Data Augmentation (Opcional pero Recomendado)** 📊
```bash
python data_augmentation.py
```
**¿Qué hace?**
- Aumenta el dataset 20x (25 → 500 secuencias)
- Aplica transformaciones: ruido, desplazamiento temporal, variaciones de velocidad
- Previene overfitting con pocos datos

#### **Paso 5: Entrenamiento del Modelo LSTM** 🧠
```bash
# Modo automático (recomendado)
python train_lstm_model.py

# Forzar dataset original (cuando tengas muchos videos)
python train_lstm_model.py --dataset original --no-augmentation

# Personalizar entrenamiento
python train_lstm_model.py --epochs 50 --batch-size 32
```
**¿Qué hace?**
- Entrena modelo LSTM con arquitectura optimizada
- Detecta automáticamente qué dataset usar
- Aplica técnicas anti-overfitting (dropout, early stopping, regularización)

#### **Paso 6: Usar la Aplicación** 🎯
```bash
python app_holistic_simple.py
```
**¿Qué hace?**
- Aplicación en tiempo real con cámara web
- Buffer de secuencias de 30 frames
- Predicciones suavizadas y estables

### 🔧 Pipeline Automatizado

#### **🤖 Pipeline LSTM (Nuevo - Recomendado)**
```bash
# Ejecutar todo el pipeline LSTM completo
python lstm_pipeline.py

# Ejecutar pasos específicos del pipeline LSTM
python lstm_pipeline.py --step 1    # Extracción de frames
python lstm_pipeline.py --step 2    # Procesamiento de secuencias
python lstm_pipeline.py --step 3    # Data augmentation
python lstm_pipeline.py --step 4    # Entrenamiento LSTM
python lstm_pipeline.py --step 5    # Verificar aplicación lista
```

#### **🔧 Pipeline General (Legacy)**
```bash
# Ejecutar todo el pipeline general
python lsa_pipeline.py

# Ejecutar pasos específicos
python lsa_pipeline.py --step frames      # Extraer frames
python lsa_pipeline.py --step sequences   # Procesar secuencias
python lsa_pipeline.py --step train       # Entrenar modelo
```

> **🎯 Diferencia**: `lstm_pipeline.py` está optimizado específicamente para el flujo LSTM con data augmentation automático.

## 📁 Estructura del Proyecto

```
interprete-lenguaje-de-senias/
├── 🎯 Scripts Principales (LSTM Pipeline)
│   ├── video_collector.py          # Grabación automática de videos
│   ├── frame_extractor.py          # Videos → Frames (30 por video)
│   ├── sequence_data_processor.py  # Frames → Secuencias de landmarks
│   ├── data_augmentation.py        # Aumento artificial de datos
│   ├── train_lstm_model.py         # Entrenamiento modelo LSTM
│   ├── app_holistic.py            # Aplicación en tiempo real
│   └── test_lstm_pipeline.py       # Pruebas del sistema completo
│
├── 🤖 Pipelines Automatizados
│   ├── lstm_pipeline.py            # Pipeline LSTM completo (NUEVO)
│   └── lsa_pipeline.py             # Pipeline general (legacy)
│
├── 🔧 Scripts de Soporte
│   ├── test_lighting_effect.py     # Pruebas de iluminación
│   └── train_holistic_model.py     # Modelo holístico (legacy)
│
├── 🧠 model/
│   ├── lstm_sign_recognizer.py     # Reconocedor LSTM principal
│   ├── holistic_sign_recognizer.py # Reconocedor holístico (legacy)
│   └── sign_recognizer.py          # Modelo básico (legacy)
│
├── 📊 data/
│   └── signs.json                  # Definiciones de señas
│
├── 📁 Directorios Generados
│   ├── Videos/                     # Videos grabados por seña
│   ├── processed_frames/           # Frames extraídos (30 por video)
│   ├── models/                     # Modelos entrenados y datasets
│   │   ├── sequence_dataset.pkl    # Dataset original
│   │   ├── sequence_dataset_augmented.pkl # Dataset aumentado
│   │   ├── lsa_lstm_model.h5       # Modelo LSTM entrenado
│   │   └── lsa_lstm_model_metadata.json # Metadatos del modelo
│   └── results/                    # Gráficos y análisis
│
└── ⚙️ Setup
    ├── requirements.txt
    ├── setup_env_gitbash.sh
    ├── setup_env.bat
    └── setup_env.sh
```

## 🧠 Arquitectura del Modelo LSTM

### **Especificaciones Técnicas**
- **Entrada**: Secuencias de 30 frames × 1662 keypoints
- **Arquitectura**: LSTM bidireccional de 2 capas
- **Keypoints**: Pose (132) + Cara (1404) + Manos (126) = 1662 total
- **Salida**: Clasificación multiclase (softmax)

### **Estructura del Modelo**
```python
Sequential([
    LSTM(64, return_sequences=True, bidirectional=True),
    Dropout(0.5),
    LSTM(128, return_sequences=False, bidirectional=True), 
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### **Técnicas Anti-Overfitting**
- ✅ **Dropout**: 0.5 en capas LSTM
- ✅ **Regularización L2**: 0.01 y 0.001
- ✅ **Early Stopping**: Paciencia de 15 épocas
- ✅ **ReduceLROnPlateau**: Factor 0.5, paciencia 8
- ✅ **Data Augmentation**: 20x aumento de datos

## 📊 Recomendaciones de Dataset

### **🎯 Cantidad de Videos Recomendada**

| Nivel | Videos por Seña | Total (5 señas) | Resultado Esperado |
|-------|----------------|-----------------|-------------------|
| **Mínimo** | 50-100 | 250-500 | Funciona básicamente |
| **Recomendado** | 200-500 | 1000-2500 | Buen rendimiento |
| **Óptimo** | 1000+ | 5000+ | Excelente precisión |
| **Tu actual** | 5 | 25 | ❌ **Requiere data augmentation** |

### **🎬 Variabilidad Necesaria**

#### **Prioridad Alta:**
- 🥇 **Diferentes personas** (5-10 mínimo)
- 🥈 **Diferentes velocidades** (lenta, normal, rápida)
- 🥉 **Diferentes condiciones de luz** (natural, artificial)

#### **Prioridad Media:**
- 📐 **Diferentes ángulos** (frontal, ligeramente lateral)
- 👕 **Diferentes ropa** (evitar que el modelo se enfoque en colores)
- 🏠 **Diferentes fondos** (variedad de backgrounds)

### **⚠️ Indicadores de Problemas**

#### **Overfitting:**
- Accuracy entrenamiento > 95%, Validation accuracy < 70%
- Loss de entrenamiento muy bajo, validation loss alto
- **Solución**: Más data augmentation, más dropout

#### **Underfitting:**
- Ambas accuracies < 60% y estables
- Loss no mejora después de muchas épocas
- **Solución**: Modelo más complejo, menos regularización

#### **Dataset Pequeño:**
- Alta varianza entre epochs
- Resultados muy diferentes en cada entrenamiento
- **Solución**: Data augmentation, más videos

## 🔧 Configuración Avanzada

### **Argumentos de `train_lstm_model.py`**
```bash
# Usar dataset original (cuando tengas muchos videos)
python train_lstm_model.py --dataset original

# Usar dataset aumentado (recomendado para pocos videos)
python train_lstm_model.py --dataset augmented

# Modo automático (detecta qué usar)
python train_lstm_model.py --dataset auto

# Personalizar entrenamiento
python train_lstm_model.py --epochs 50 --batch-size 32

# Modo no interactivo (para scripts)
python train_lstm_model.py --no-augmentation
```

### **Parámetros de Data Augmentation**
```python
# En data_augmentation.py puedes modificar:
augmentation_factor = 20    # Factor de aumento (20x más datos)
noise_factor = 0.01        # Cantidad de ruido gaussiano
shift_range = 3            # Desplazamiento temporal máximo
speed_range = (0.8, 1.2)   # Variación de velocidad
```

### 📝 Agregar Nuevas Señas

#### **Método Automatizado (Recomendado)**
```bash
# 1. Grabar videos de nuevas señas
python video_collector.py

# 2. Ejecutar pipeline completo para reentrenar
python lstm_pipeline.py
```

#### **Método Manual**
1. **Grabar videos**: `python video_collector.py`
2. **Extraer frames**: `python frame_extractor.py`
3. **Procesar secuencias**: `python sequence_data_processor.py`
4. **Data augmentation**: `python data_augmentation.py`
5. **Entrenar modelo**: `python train_lstm_model.py`

## 🔍 Pruebas y Verificación

### **Verificar que Todo Funciona**

#### **Verificación Automática (Recomendado)**
```bash
# El pipeline LSTM incluye verificación automática
python lstm_pipeline.py --step 5
```

#### **Verificación Manual**
```bash
python test_lstm_pipeline.py
```
**¿Qué verifica?**
- ✅ Frames procesados existentes
- ✅ Dataset de secuencias creado
- ✅ Modelo LSTM entrenado
- ✅ Reconocedor funcionando

### **Probar Efectos de Iluminación**
```bash
python test_lighting_effect.py
```
**¿Para qué sirve?**
- Mide cómo afecta la luz a la detección de landmarks
- Te dice las mejores condiciones para grabar

## 🚨 Solución de Problemas

### **❌ Error Común: "Frases encontradas: []"**

**Síntoma:**
```
=== PROCESADOR DE SECUENCIAS LSA ===
Frases encontradas: []
Error: No se pudieron procesar datos
```

**Causa:** El `sequence_data_processor.py` busca frames en `processed_frames/` pero no existen.

**Solución Rápida (Recomendada):**
```bash
# Usar el pipeline automatizado que maneja este error
python lstm_pipeline.py
```

**Solución Manual:**
```bash
# 1. Verificar que tienes videos
ls videos/  # Debe mostrar carpetas HOLA, CHAU, etc.

# 2. Extraer frames de los videos PRIMERO
python frame_extractor.py

# 3. LUEGO procesar secuencias
python sequence_data_processor.py
```

### **Error: "Modelo LSTM no encontrado"**

**Solución Rápida:**
```bash
# Pipeline automatizado (maneja todos los pasos)
python lstm_pipeline.py
```

**Solución Manual:**
```bash
python frame_extractor.py
python sequence_data_processor.py
python train_lstm_model.py
```

### **Error: "Dataset no encontrado"**

**Solución Rápida:**
```bash
# Pipeline automatizado desde el paso 2
python lstm_pipeline.py --step 2
```

**Solución Manual:**
```bash
python frame_extractor.py
python sequence_data_processor.py
```

### **Error: "No se encontraron videos"**

**Solución:**
```bash
# 1. Grabar videos primero
python video_collector.py

# 2. Luego ejecutar pipeline completo
python lstm_pipeline.py
```

### **Error: "No se encontraron frames procesados"**

**Solución Rápida:**
```bash
# Pipeline desde el paso 1
python lstm_pipeline.py --step 1
```

**Solución Manual:**
```bash
python frame_extractor.py  # Videos → Frames
python sequence_data_processor.py  # Frames → Dataset
```

### **Overfitting Detectado (Accuracy train >> validation)**
```bash
# Solución 1: Usar data augmentation
python data_augmentation.py
python train_lstm_model.py --dataset augmented

# Solución 2: Grabar más videos
python video_collector.py  # Grabar 50+ por seña
python train_lstm_model.py --dataset original
```

### **Underfitting (Ambas accuracies bajas)**
```bash
# Solución: Entrenar más tiempo o cambiar parámetros
python train_lstm_model.py --epochs 200 --batch-size 8
```

### **Predicciones Inestables en Tiempo Real**
- **Causa**: Buffer de secuencias no lleno
- **Solución**: Espera a que se llene el buffer (30 frames)
- **Indicador**: Barra de progreso en la aplicación

### **Problemas de Memoria**
```bash
# Reducir batch size
python train_lstm_model.py --batch-size 8

# O usar menos data augmentation
# Editar data_augmentation.py: augmentation_factor = 10
```

## 🎯 Guía de Inicio para Nuevos Usuarios

### **👶 Primera Vez Usando el Sistema**

1. **📥 Instalar dependencias**:
   ```bash
   # Windows
   setup_env.bat
   
   # Linux/Mac
   ./setup_env.sh
   ```

2. **📹 Grabar tus primeros videos**:
   ```bash
   python video_collector.py
   ```
   - Graba al menos 10-20 videos por seña
   - Usa buena iluminación
   - Mantén las manos visibles

3. **🤖 Ejecutar pipeline completo**:
   ```bash
   python lstm_pipeline.py
   ```
   - El sistema te guiará paso a paso
   - Maneja errores automáticamente
   - Te dice exactamente qué hacer si algo falla

4. **🎮 Probar tu traductor**:
   ```bash
   python app_holistic.py
   ```
   - Presiona "▶️ Iniciar Reconocimiento"
   - Realiza señas frente a la cámara
   - ¡Disfruta tu traductor de LSA!

### **🔧 Para Desarrolladores Avanzados**

- **Modificar parámetros**: Edita `data_augmentation.py` y `train_lstm_model.py`
- **Agregar señas**: Usa `video_collector.py` + `lstm_pipeline.py`
- **Experimentar**: Prueba diferentes arquitecturas en `train_lstm_model.py`
- **Depurar**: Usa `test_lstm_pipeline.py` para diagnósticos

## 🚀 Próximas Mejoras

- [ ] **Más señas**: Expandir vocabulario LSA
- [ ] **Interfaz web**: Flask/FastAPI para uso remoto
- [ ] **Modelo Transformer**: Para mayor precisión
- [ ] **Dataset público**: Compartir datos de LSA
- [ ] **Métricas avanzadas**: Evaluación automática
- [ ] **Multi-persona**: Reconocimiento simultáneo
- [ ] **Pipeline en la nube**: Entrenamiento distribuido

## 📚 Notas Técnicas

### **Arquitectura del Sistema**
Este sistema implementa un enfoque de **una etapa** optimizado:
- **Entrada**: Video de señas LSA
- **Procesamiento**: Extracción de landmarks holísticos
- **Modelo**: LSTM bidireccional para secuencias
- **Salida**: Clasificación directa de señas

### **Diferencias con Sistemas Tradicionales**
- ✅ **Landmarks vs Píxeles**: Más eficiente y robusto
- ✅ **Secuencias vs Frames**: Captura el movimiento temporal
- ✅ **LSTM vs CNN**: Mejor para patrones temporales
- ✅ **Holístico vs Solo Manos**: Más información contextual

### **Rendimiento Esperado**
- **Con 25 videos + augmentation**: ~70-80% accuracy
- **Con 100+ videos por seña**: ~85-95% accuracy
- **Con 500+ videos por seña**: ~95%+ accuracy

### **Comandos Más Usados**

```bash
# 🚀 Comando más importante (para principiantes)
python lstm_pipeline.py

# 📹 Grabar más videos
python video_collector.py

# 🎮 Usar el traductor
python app_holistic.py

# 🔍 Verificar que todo funciona
python lstm_pipeline.py --step 5

# 🛠️ Solo entrenar modelo (si ya tienes datos)
python lstm_pipeline.py --step 4
```

---

**💡 ¿Tienes problemas?** Ejecuta `python lstm_pipeline.py` - el sistema detecta automáticamente qué falta y te guía para solucionarlo.

**🎉 ¿Todo funcionó?** ¡Comparte tu experiencia y ayuda a mejorar el proyecto!