# API Multimodal  – Detección de Botaderos a Cielo Abiero

Este proyecto implementa una **API basada en FastAPI y Docker** para inferencia de un **modelo multimodal** que combina imágenes satelitales y datos atmosféricos con el fin de detectar la presencia de botaderos a cielo abierto en Guatemala.

La arquitectura está basada en el trabajo **AerialWaste** de Torres & Fraternali (2023), y reutiliza el modelo visual preentrenado sobre dicho conjunto de datos.  
Esta API permite enviar imágenes (como archivos binarios o en base64) junto con cuatro variables atmosféricas para obtener la **probabilidad y etiqueta predicha** de la presencia de un vertedero.

---

## Referencia

Si utilizas esta API o el modelo en tus trabajos académicos, cita el artículo original:
```
@article{torres2023aerialwaste,
  title={AerialWaste dataset for landfill discovery in aerial and satellite images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={63},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```


---

## 1. Arquitectura del Modelo

El modelo multimodal integra dos fuentes de información:

1. **Rama Visual (CNN con ResNet50-FPN):**
   - Basada en la arquitectura `CAM_PRED` de *AerialWaste*.
   - Extrae la probabilidad sigmoide de que una imagen contenga un vertedero.
   - Se usa esta probabilidad (no las features internas) como entrada a la fusión multimodal.

2. **Rama Atmosférica (MLP):**
   - Recibe cuatro variables: **NO₂**, **CO**, **PM₂.₅** y **PM₁₀**.
   - Genera una representación latente de 16 dimensiones.

3. **Fusión Multimodal:**
   - Concatena la probabilidad visual (1D) con las features atmosféricas (16D).
   - Pasa por un perceptrón multicapa (MLP) de tres capas: 256 → 64 → 1.
   - Devuelve un *logit* cuya sigmoide corresponde a la probabilidad final.

El modelo se construye con:

```python
from architecture.multimodal_arch import build_model
model = build_model(visual_ckpt_path, visual_microbatch=2)
```

## 2. Requisitos
Archivos de modelo

Debes tener los siguientes archivos entrenados previamente:

- `checkpoint.pth` – Pesos del modelo visual (rama base AerialWaste)

- `best_phase2.pth` – Pesos finales del modelo multimodal

- `atmos_zscore.npz` – Estadísticos de normalización (media y desviación)

Estos se montarán dentro del contenedor en `/models`.

Dependencias principales

- Python 3.10+

- FastAPI, Uvicorn

- PyTorch, TorchVision

- Pillow, NumPy

Todas se instalan automáticamente dentro del contenedor mediante `requirements.txt`.

## 3. Configuración del Entorno

Crea un archivo `.env` en el directorio raíz (o usa el `.env.example`) con las siguientes variables:
```
VISUAL_CKPT_PATH=/models/checkpoint.pth
FINAL_WEIGHTS=/models/best_phase2.pth
ATMOS_Z_PATH=/models/atmos_zscore.npz
THRESH=0.5
VISUAL_MICROBATCH=2
```

- `THRESH`: umbral de clasificación (por defecto 0.5).

- `VISUAL_MICROBATCH`: tamaño de microbatch para evitar errores de memoria.

## 4. Construcción del Contenedor Docker

Desde la raíz del proyecto:
```
docker build -t ecomentat-api .
```

Luego, ejecuta el contenedor montando los archivos del modelo y el .env:

```
docker run --rm -p 8000:8000 \
  --env-file ./.env \
  -v /ruta/absoluta/a/modelos:/models:ro \
  ecomentat-api
```

Esto levantará el servicio en:
```
http://localhost:8000
```

## 5. Uso de la API

La API expone tres endpoints principales:

### 1. Comprobación de vida
```
GET /live
```

Respuesta:
```
{"ok": true}
```


### 2. Predicción en formato JSON (imagen base64)
```
POST /predict
```

**Body (JSON):**
```
{
  "image_base64": "<cadena_base64_de_la_imagen>",
  "atmos": {
    "NO2": 10.0,
    "CO": 0.3,
    "PM2.5": 25.0,
    "PM10": 40.0
  }
}
```

Respuesta:
```
{
  "prob": 0.7324,
  "label": 1
}
```

### 3. Predicción en formato multipart/form-data
```
POST /predict-multipart
```

Campos:

- `file`: imagen (PNG/JPEG)

- `NO2`, `CO`, `PM2.5`, `PM10`: valores numéricos

Ejemplo en Python:
```
import requests
from PIL import Image
from io import BytesIO

img = Image.new("RGB",(224,224),(120,180,200))
buf = BytesIO(); img.save(buf, format="PNG")

files = {"file": ("test.png", buf.getvalue(), "image/png")}
data = {"NO2": "10.0", "CO": "0.3", "PM2.5": "25.0", "PM10": "40.0"}

r = requests.post("http://localhost:8000/predict-multipart", files=files, data=data)
print(r.json())
```


### 6. Pruebas dentro del contenedor

El proyecto incluye un pequeño conjunto de pruebas automáticas con pytest para validar que la API responde correctamente.

Ejecuta las pruebas directamente dentro del contenedor:
```
docker run --rm \
  --env-file ./.env \
  -v /ruta/absoluta/a/modelos:/models:ro \
  ecomentat-api \
  python -m pytest -q
  ```

### 7. Notas finales

Si deseas usar GPU, sustituye la imagen base por una imagen NVIDIA CUDA y ejecuta con `--gpus all`.

Para desplegar en producción, se recomienda usar un reverse proxy (NGINX) y limitar el tamaño máximo de archivo subido.

Esta API está optimizada para inferencia, no para entrenamiento.

Autoría

Desarrollado como parte del proyecto de detección de botaderos a cielo abierto en Guatemala, extendiendo el trabajo de:

Torres, R. N., & Fraternali, P. (2023).
AerialWaste dataset for landfill discovery in aerial and satellite images.
Scientific Data, 10(1), 63.
Nature Publishing Group UK London.

