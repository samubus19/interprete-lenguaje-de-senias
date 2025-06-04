#!/bin/bash

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

echo "Entorno virtual configurado exitosamente!"
echo "Para activar el entorno virtual, ejecuta: source venv/bin/activate" 