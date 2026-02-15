# 1-Extraction: Extracción de keypoints desde videos

Scripts para extraer **keypoints** (puntos clave) de videos de señas con MediaPipe Holistic y generar la estructura que consume el Colab de entrenamiento (`Copia_de_cacic_escuela_de_inform_tica_demo.ipynb`).

## ¿Landmarks o keypoints?

En este proyecto **se usan como sinónimos**: son las posiciones (x, y) más un valor de confianza que MediaPipe detecta en cada frame (cuerpo, manos). El Colab habla de "keypoints"; MediaPipe los llama "landmarks". La salida es la misma: arrays `(num_frames, 75, 3)` con 75 puntos por frame.

## Formato de salida (compatible con el Colab)

- **543 keypoints por frame**: 33 pose (cuerpo) + 468 cara (MediaPipe Face) + 21 mano izquierda + 21 mano derecha.
- Cada keypoint: `(x, y, confidence)`. Coordenadas normalizadas en [0, 1].
- Archivos `.npy` con forma `(num_frames, 543, 3)`.
- `labels.csv` con columnas `ID` y `Label` (ID = nombre del .npy sin extensión, Label = palabra, ej. AGUA, CHAU).

## Estructura de videos de entrada

Se espera una carpeta (por defecto `videos/`) con una subcarpeta por palabra, y dentro los videos numerados:

```
videos/
├── AGUA/
│   ├── 1.mp4
│   ├── 2.mp4
│   └── ...
├── CHAU/
│   ├── 1.mp4
│   └── ...
└── ...
```

Formatos admitidos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

## ¿Hace falta extraer frames a disco?

No. Los scripts leen el video con OpenCV (`cv2.VideoCapture`) y pasan cada frame en memoria a MediaPipe. No es necesario exportar frames a imágenes; todo el proceso es video → frames en memoria → detección → array por video → guardar `.npy`.

## Constantes (en `process_videos.py`)

Podés editar al inicio del archivo:

- **`VIDEOS_DIR`**: ruta donde están los videos (p. ej. `"videos"` o `"C:/ruta/mis_videos"`). Es el valor por defecto si no pasás la carpeta por línea de comandos.
- **`MAX_FOLDERS`**: número de carpetas (palabras) a procesar, o `None` para todas. Por defecto `3` para probar solo las primeras 3; cuando funcione, poné `None` o ejecutá con `--max-folders 0` para procesar todas.

## Uso

### Opción 1: Procesar (por defecto usa VIDEOS_DIR y MAX_FOLDERS)

Desde la **raíz del proyecto**:

```bash
python train-data/1-extraction/process_videos.py -o train-data/extraction_output
```

Usa la ruta definida en `VIDEOS_DIR` y el límite en `MAX_FOLDERS`. Para indicar otra carpeta de videos sin cambiar el archivo:

```bash
python train-data/1-extraction/process_videos.py "C:/ruta/mis_videos" -o train-data/extraction_output
```

Para procesar **todas** las carpetas (ignorar `MAX_FOLDERS`): `--max-folders 0`.

- `-o train-data/extraction_output`: donde se creará `data/train/poses/` y `data/train/labels.csv`.

Para usar esos datos en el Colab, apunta `input_path` a la carpeta que contiene `data/` (en el ejemplo: `train-data/extraction_output` o subir esa carpeta a Drive y usar su ruta).

### Opción 2: Un solo video

```bash
python train-data/1-extraction/extract_keypoints.py path/al/video.mp4 -o salida.npy
```

### Opciones útiles de `process_videos.py`

- `--max-folders N`: procesar solo las primeras N carpetas; `0` = todas (por defecto usa la constante `MAX_FOLDERS`).
- `--skip-existing`: no reprocesar videos que ya tengan su `.npy` generado.
- `--min-detection`, `--min-tracking`: umbrales de MediaPipe (por defecto 0.5).

## Archivos en esta carpeta

| Archivo | Descripción |
|---------|-------------|
| `extract_keypoints.py` | Extrae keypoints de un video y opcionalmente guarda un `.npy`. |
| `process_videos.py` | Recorre `videos/PALABRA/*.mp4`, genera todos los `.npy` y `labels.csv`. |
| `utils.py` | Funciones auxiliares (extensiones de video, rutas). |

## Dependencias

- Python 3.8+
- `opencv-python`, `mediapipe`, `numpy`, `pandas`

Si en el proyecto ya usás MediaPipe (por ej. en `video_collector.py`), las mismas dependencias sirven.
