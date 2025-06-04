# Intérprete de Lenguaje de Señas Argentino

Esta aplicación permite interpretar el lenguaje de señas argentino en tiempo real, convirtiendo señas a texto y viceversa.

## Requisitos

- Python 3.8 o superior
- Cámara web
- Dependencias listadas en `requirements.txt`

## Instalación

### Configuración del Entorno Virtual

#### En Git Bash (Windows):
```bash
# Dar permisos de ejecución al script
chmod +x setup_env_gitbash.sh

# Ejecutar el script de configuración
./setup_env_gitbash.sh

# O manualmente:
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

#### En Windows (Command Prompt):
```bash
# Ejecutar el script de configuración
setup_env.bat

# O manualmente:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### En Linux/Mac:
```bash
# Dar permisos de ejecución al script
chmod +x setup_env.sh

# Ejecutar el script de configuración
./setup_env.sh

# O manualmente:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

1. Activar el entorno virtual:
   - Git Bash: `source venv/Scripts/activate`
   - Command Prompt: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

2. Recolectar datos de entrenamiento:
```bash
python collect_data.py
```

3. Entrenar el modelo:
```bash
python train_model.py
```

4. Ejecutar la aplicación:
```bash
python app.py
```

## Estructura del Proyecto

```
.
├── app.py                 # Aplicación principal
├── collect_data.py        # Script para recolectar datos
├── train_model.py         # Script para entrenar el modelo
├── requirements.txt       # Dependencias del proyecto
├── setup_env_gitbash.sh   # Script de configuración para Git Bash
├── setup_env.bat         # Script de configuración para Windows
├── setup_env.sh          # Script de configuración para Linux/Mac
├── data/
│   ├── signs.json        # Definición de señas
│   └── collected/        # Datos recolectados
├── model/
│   ├── sign_recognizer.py # Modelo de reconocimiento
│   └── data_collector.py  # Recolector de datos
└── models/               # Modelos entrenados
```

## Características

- Reconocimiento de señas en tiempo real
- Interfaz gráfica intuitiva
- Soporte para múltiples señas simultáneas
- Visualización de puntos de referencia de las manos
- Sistema de recolección de datos
- Entrenamiento y evaluación del modelo

## Notas

Esta es una versión inicial de la aplicación. El reconocimiento de señas se irá mejorando con el tiempo y la recopilación de más datos de entrenamiento. 