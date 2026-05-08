# 🎨 Florence-2 Auto Caption Pipeline

Pipeline de captioning automático para datasets de entrenamiento de modelos generativos de imagen (LoRA / fine-tuning). Utiliza **Florence-2-large** de Microsoft ejecutado en local para generar descripciones detalladas de imágenes, listas para usar con herramientas como [ai-toolkit](https://github.com/ostris/ai-toolkit) o Kohya.

Desarrollado como parte de un TFG sobre pipelines de entrenamiento de modelos generativos orientados a la producción de imágenes en estilos predeterminados (anime).

---

## ¿Qué hace?

Dado un directorio con imágenes, el script genera automáticamente un archivo `.txt` por imagen con una descripción detallada en inglés, añadiendo al inicio un **trigger word** personalizable. Estos archivos `.txt` son exactamente el formato que esperan herramientas como ai-toolkit o Kohya para el entrenamiento de LoRAs.

**Ejemplo de output generado:**
```
saoStyle, The image is a close-up of a young girl's face. She has blonde hair 
styled in two pigtails on top of her head. She is wearing a white and yellow 
armored chest piece. A small blue dragon is perched on top of her head...
```

---

## Requisitos

- Python 3.10 o superior
- Windows, Linux o macOS
- GPU opcional (funciona en CPU, aunque más lento)
- Cuenta en [Hugging Face](https://huggingface.co) con token de acceso

### Dependencias Python

```bash
pip install transformers torch pillow einops timm huggingface_hub
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/florence2-caption-pipeline.git
cd florence2-caption-pipeline
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install transformers torch pillow einops timm huggingface_hub
```

### 4. Instalar stub de flash_attn (necesario en Windows y entornos CPU)

Florence-2 declara `flash_attn` como dependencia, pero esta librería requiere compilación con CUDA y no funciona en CPU ni en Windows. Para evitar el error de instalación, se crea un paquete stub que satisface el import sin necesidad de instalación real:

```bash
python install_flash_attn_stub.py
```

> **¿Por qué es necesario esto?** `transformers` realiza un análisis estático del código de Florence-2 antes de ejecutarlo y lanza un `ImportError` si no encuentra `flash_attn`, aunque el modelo se ejecute en modo `eager` (CPU) donde flash attention nunca se usa. El stub resuelve esto de forma limpia sin modificar los archivos del modelo.

### 5. Login en Hugging Face

```bash
huggingface-cli login
```

O configurar la variable de entorno:
```bash
# Windows
$env:HUGGINGFACE_HUB_TOKEN="tu_token"

# Linux / macOS
export HUGGINGFACE_HUB_TOKEN="tu_token"
```

---

## Uso

### Uso básico

```bash
python generate_captions.py --input-folder ./mi_dataset
```

### Con trigger word personalizado

```bash
python generate_captions.py --input-folder ./mi_dataset --trigger-word "miEstilo"
```

### Guardar captions en carpeta distinta a las imágenes

```bash
python generate_captions.py --input-folder ./imagenes --output-folder ./captions
```

### Forzar uso de GPU

```bash
python generate_captions.py --input-folder ./mi_dataset --device cuda
```

### Todos los parámetros disponibles

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `--input-folder` | `./dataset_sao` | Carpeta con las imágenes a procesar |
| `--output-folder` | misma que input | Carpeta donde guardar los `.txt` |
| `--trigger-word` | `saoStyle` | Palabra clave al inicio de cada caption |
| `--device` | auto-detectado | `cpu` o `cuda` |
| `--extensions` | `.jpg,.jpeg,.png,.bmp,.webp` | Extensiones de imagen a procesar |

---

## Estructura del repositorio

```
florence2-caption-pipeline/
├── generate_captions.py       # Script principal de captioning
├── install_flash_attn_stub.py # Instalador del stub de flash_attn
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

---

## Cómo funciona internamente

1. **Descarga del modelo**: Florence-2-large (~1.5GB) se descarga automáticamente desde Hugging Face en la primera ejecución y se cachea localmente.
2. **Carga del modelo**: Se carga con `attn_implementation="eager"` para compatibilidad total con CPU.
3. **Generación de captions**: Para cada imagen se ejecuta Florence-2 con el task prompt `<MORE_DETAILED_CAPTION>`, que genera descripciones detalladas de escena, personajes, colores e iluminación.
4. **Post-procesado**: Se añade el trigger word al inicio y se guarda el resultado en un archivo `.txt` con el mismo nombre que la imagen.

---

## Rendimiento estimado

| Hardware | Tiempo por imagen |
|---|---|
| CPU (i5/i7 moderno) | 2–5 minutos |
| GPU (RTX 3060+) | 5–15 segundos |

Para datasets grandes se recomienda encarecidamente usar GPU.

---

## Compatibilidad

| Sistema | Soporte |
|---|---|
| Windows 10/11 | ✅ (requiere stub flash_attn) |
| Linux | ✅ |
| macOS (Apple Silicon) | ✅ (device cpu) |

---

## Contexto del proyecto

Este script forma parte de un pipeline completo de entrenamiento de modelos generativos para estilos de imagen predeterminados, desarrollado como Trabajo de Fin de Grado. El pipeline completo incluye:

- **Captioning automático** (este repositorio): generación de descriptores para el dataset
- **Entrenamiento de LoRA**: fine-tuning de Flux.1-dev con ai-toolkit
- **Evaluación**: generación de muestras y análisis de resultados

---

## Licencia

MIT License — libre para uso académico y personal.