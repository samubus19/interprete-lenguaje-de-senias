@echo off

REM Crear entorno virtual
python -m venv venv

REM Activar entorno virtual
call venv\Scripts\activate

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar dependencias
pip install -r requirements.txt

echo Entorno virtual configurado exitosamente!
echo Para activar el entorno virtual, ejecuta: venv\Scripts\activate 