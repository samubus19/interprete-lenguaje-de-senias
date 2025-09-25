# Traductor de Lengua de Señas Argentina (LSA) a Texto

Sistema completo de traducción de LSA a texto natural utilizando arquitectura de dos etapas:
1. **Modelo de secuencias**: Video → Glosas (palabras individuales)
2. **Procesamiento de lenguaje natural**: Glosas → Texto natural

## Características Principales

- **Pipeline completo automatizado** para procesamiento de videos LSA
- **Extracción inteligente de frames** representativos usando clustering
- **Landmarks holísticos** con MediaPipe (manos, pose, cara)
- **Modelos de secuencias avanzados** (LSTM, GRU, Transformer)
- **Conversión glosas → texto** con múltiples métodos (OpenAI, Transformers, Reglas)
- **Interfaz de grabación automática** para recolección de datos

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

## Uso del Sistema

### Opción 1: Pipeline Completo Automatizado
```bash
# Ejecutar todo el pipeline de una vez
python lsa_pipeline.py

# Con configuración personalizada
python lsa_pipeline.py --config mi_config.json

# Crear archivo de configuración
python lsa_pipeline.py --create-config
```

### Opción 2: Paso a Paso

#### 1. Grabar Videos de Frases
```bash
python video_collector.py
```
- Graba videos automáticamente al detectar manos
- Estructura: `./videos/FRASE/1.avi`
- Modo automático o manual disponible

#### 2. Procesar Videos → Frames → Landmarks → Dataset
```bash
# Extraer 30 frames representativos por video
python frame_extractor.py

# Extraer landmarks con MediaPipe Holistic
python landmark_extractor.py

# O ejecutar ambos pasos:
python lsa_pipeline.py --step frames
python lsa_pipeline.py --step landmarks
python lsa_pipeline.py --step dataset
```

#### 3. Entrenar Modelo de Secuencias
```bash
# Entrenar modelo LSTM (por defecto)
python train_sequence_model.py

# Entrenar modelo específico
python train_sequence_model.py --model_type transformer --epochs 100

# O usando el pipeline
python lsa_pipeline.py --step train
```

#### 4. Configurar Conversión de Glosas
```bash
# Probar convertidor de glosas
python gloss_to_text.py

# O usando el pipeline
python lsa_pipeline.py --step gloss
```

#### 5. Usar la Aplicación Completa
```bash
python app.py
```

## Estructura del Proyecto

```
interprete-lenguaje-de-senias/
├── Core Scripts
│   ├── lsa_pipeline.py           # Pipeline completo automatizado
│   ├── video_collector.py        # Grabación automática de videos
│   ├── frame_extractor.py        # Extracción de frames representativos
│   ├── landmark_extractor.py     # Extracción de landmarks holísticos
│   ├── train_sequence_model.py   # Entrenamiento de modelos
│   ├── gloss_to_text.py         # Conversión glosas → texto
│   └── app.py                   # Aplicación principal
│
├── model/
│   ├── sequence_model.py        # Modelos LSTM/GRU/Transformer
│   └── sign_recognizer.py       # Modelo básico (legacy)
│
├── data/
│   └── signs.json              # Definiciones de frases y glosas
│
├── Directorios Generados
│   ├── videos/                 # Videos grabados por frase
│   ├── processed_frames/       # Frames extraídos
│   ├── landmarks_data/         # Landmarks procesados
│   ├── dataset/               # Dataset unificado
│   ├── models/                # Modelos entrenados
│   └── results/               # Resultados y análisis
│
└── Setup
    ├── requirements.txt
    ├── setup_env_gitbash.sh
    ├── setup_env.bat
    └── setup_env.sh
```

## Configuración Avanzada

### Archivo de Configuración (`lsa_config.json`)
```json
{
    "videos_dir": "videos",
    "target_frames": 30,
    "model_type": "lstm",
    "gloss_method": "rules",
    "openai_api_key": "tu-api-key-aqui"
}
```

### Métodos de Conversión de Glosas

1. **Reglas Gramaticales** (por defecto)
   - Sin dependencias externas
   - Rápido y eficiente
   - Bueno para frases simples

2. **OpenAI API** (recomendado)
   - Mejor calidad de traducción
   - Requiere API key
   - Costo por uso

3. **Transformers Locales**
   - Sin costos de API
   - Requiere más recursos
   - Personalizable

### Configurar OpenAI (Opcional)
```bash
# Opción 1: Variable de entorno
export OPENAI_API_KEY="tu-api-key"

# Opción 2: En el archivo de configuración
# Editar lsa_config.json y agregar tu API key
```

## Ejemplos de Uso

### Frases de Ejemplo Incluidas
- "YO COMPRAR CARNE" → "Yo voy a comprar carne"
- "MAMÁ COCINAR COMIDA RICA" → "Mamá cocina comida rica"
- "MAÑANA TRABAJO IR" → "Mañana voy a trabajar"
- "HERMANO PELOTA JUGAR" → "Mi hermano juega a la pelota"

### Agregar Nuevas Frases
1. Editar `data/signs.json`
2. Grabar videos con `video_collector.py`
3. Ejecutar pipeline completo

## Solución de Problemas

### Error: "No se encontró el directorio de videos"
```bash
# Crear videos primero
python video_collector.py
```

### Error: "Dataset no encontrado"
```bash
# Ejecutar pasos previos
python lsa_pipeline.py --step frames
python lsa_pipeline.py --step landmarks
python lsa_pipeline.py --step dataset
```

### Error: "Modelo de spaCy no encontrado"
```bash
python -m spacy download es_core_news_sm
```

### Problemas de Memoria
- Reducir `target_frames` en configuración
- Usar modelo `lstm` en lugar de `transformer`
- Procesar menos videos a la vez

## Próximas Mejoras

- [ ] Interfaz web con Flask/FastAPI
- [ ] Soporte para video en tiempo real
- [ ] Más modelos de NLP para glosas
- [ ] Dataset público de LSA
- [ ] Métricas de evaluación automática
- [ ] Soporte para múltiples idiomas de señas

## Notas de Desarrollo

Este sistema implementa un enfoque innovador de dos etapas para la traducción de LSA:

1. **Primera etapa**: Convierte secuencias de video a glosas usando modelos de deep learning
2. **Segunda etapa**: Transforma glosas a texto natural usando NLP

La arquitectura permite entrenar cada etapa independientemente y combinar diferentes enfoques según las necesidades específicas.