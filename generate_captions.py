#!/usr/bin/env python3
"""
Pipeline de captioning automático con Florence-2 ejecutado en local.

Genera un archivo .txt por imagen con una descripción detallada,
lista para usar en entrenamiento de LoRA con ai-toolkit o Kohya.

Uso básico:
    python generate_captions.py --input-folder ./mi_dataset

Con trigger word personalizado:
    python generate_captions.py --input-folder ./mi_dataset --trigger-word "miEstilo"

Requisitos:
    pip install transformers torch pillow einops timm huggingface_hub
    python install_flash_attn_stub.py  (necesario en Windows / CPU)
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import sys
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download


# =============================================================================
# Configuración — modifica estas variables si lo necesitas
# =============================================================================
MODEL_NAME = "microsoft/Florence-2-large"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

# Carpeta local donde se descarga y cachea el modelo (~1.5GB, solo primera vez)
LOCAL_MODEL_DIR = Path.home() / ".cache" / "huggingface" / "florence2-large"


# =============================================================================
# Carga del modelo
# =============================================================================
def load_model(device):
    print(f"Descargando/verificando modelo (solo descarga en la primera ejecución)...")
    print(f"  Modelo   : {MODEL_NAME}")
    print(f"  Caché    : {LOCAL_MODEL_DIR}")
    print(f"  Dispositivo: {device}\n")

    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False
    )

    print(f"Cargando modelo desde: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # compatible con CPU, no requiere flash_attn
        _fast_init=False
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✓ Modelo cargado.\n")
    return model, processor


# =============================================================================
# Generación de caption para una imagen
# =============================================================================
def generate_caption(image_path, model, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=TASK_PROMPT,
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3
            )

        result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            result,
            task=TASK_PROMPT,
            image_size=(image.width, image.height)
        )
        return parsed[TASK_PROMPT].strip()

    except Exception as e:
        print(f"  Error procesando {image_path.name}: {e}")
        return None


# =============================================================================
# Pipeline principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Captioning automático con Florence-2 para datasets de entrenamiento LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python generate_captions.py --input-folder ./mi_dataset
  python generate_captions.py --input-folder ./imagenes --trigger-word "miEstilo"
  python generate_captions.py --input-folder ./imagenes --output-folder ./captions --device cuda
        """
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,                          # obligatorio: el usuario debe especificarlo
        help="Carpeta con las imágenes a procesar"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Carpeta donde guardar los .txt (por defecto: misma carpeta que --input-folder)"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default="myStyle",                      # genérico, el usuario lo cambia por el suyo
        help="Palabra clave a añadir al inicio de cada caption (por defecto: myStyle)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Dispositivo a usar: cpu o cuda (por defecto: auto-detectado)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Extensiones de imagen a procesar, separadas por comas"
    )
    args = parser.parse_args()

    # Detectar dispositivo automáticamente si no se especifica
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder) if args.output_folder else input_folder

    if not input_folder.exists():
        print(f"❌ Carpeta no encontrada: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    valid_exts = set(args.extensions.lower().split(","))
    image_files = sorted([
        f for f in input_folder.iterdir()
        if f.suffix.lower() in valid_exts
    ])

    if not image_files:
        print(f"❌ No se encontraron imágenes en: {input_folder}")
        print(f"   Extensiones buscadas: {valid_exts}")
        sys.exit(1)

    print(f"Carpeta entrada  : {input_folder}")
    print(f"Carpeta salida   : {output_folder}")
    print(f"Trigger word     : {args.trigger_word}")
    print(f"Imágenes         : {len(image_files)}")
    print("=" * 80)

    model, processor = load_model(device)

    success_count = 0
    failed_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {image_path.name}")
        caption = generate_caption(image_path, model, processor, device)

        if caption:
            final_caption = f"{args.trigger_word}, {caption}"
            output_path = output_folder / f"{image_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_caption)
            print(f"  ✓ {final_caption[:120]}...\n")
            success_count += 1
        else:
            print(f"  ✗ Error generando caption\n")
            failed_count += 1

    print("=" * 80)
    print(f"\nResumen — Exitosas: {success_count} | Fallidas: {failed_count}")
    print(f"Captions guardados en: {output_folder}")


if __name__ == "__main__":
    main()