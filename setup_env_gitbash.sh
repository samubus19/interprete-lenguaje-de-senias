#!/bin/bash

# Verificar si Python está instalado
if ! command -v python &> /dev/null; then
    echo "Error: Python no está instalado o no está en el PATH"
    exit 1
fi

# Crear entorno virtual
echo "Creando entorno virtual..."
python -m venv venv

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/Scripts/activate

# Verificar si la activación fue exitosa
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No se pudo activar el entorno virtual"
    exit 1
fi

# Actualizar pip
echo "Actualizando pip..."
python -m pip install --upgrade pip

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

echo "¡Entorno virtual configurado exitosamente!"
echo "Para activar el entorno virtual, ejecuta: source venv/Scripts/activate"
echo "Para desactivar el entorno virtual, ejecuta: deactivate" 