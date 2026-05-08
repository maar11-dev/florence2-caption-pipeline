#!/usr/bin/env python3
"""
Instalador del stub de flash_attn para compatibilidad CPU/Windows.

Florence-2 declara flash_attn como dependencia pero solo la usa con
attn_implementation='flash_attention_2'. En modo 'eager' (CPU) nunca
se llama, pero transformers hace un check estático que falla si el
paquete no está instalado.

Este script crea un paquete stub que satisface el import sin necesidad
de compilar flash_attn con CUDA.
"""

from pathlib import Path
import site
import sys


def install_stub():
    # Buscar site-packages del entorno activo
    site_packages = site.getsitepackages()
    if not site_packages:
        print("❌ No se encontró el directorio site-packages")
        sys.exit(1)

    sp = Path(site_packages[0])
    fa_dir = sp / "flash_attn"

    if fa_dir.exists():
        print(f"ℹ️  El stub ya existe en: {fa_dir}")
        print("   Si quieres reinstalarlo, borra la carpeta manualmente y vuelve a ejecutar.")
        return

    # Crear paquete stub
    fa_dir.mkdir(parents=True, exist_ok=True)

    (fa_dir / "__init__.py").write_text(
        '# flash_attn stub — compatibilidad CPU/Windows\n'
        '# Las funciones reales nunca se llaman en modo eager (CPU)\n\n'
        'def flash_attn_func(*args, **kwargs):\n'
        '    raise RuntimeError("flash_attn real no está instalado. Usa attn_implementation=\'eager\' para CPU.")\n\n'
        'def flash_attn_varlen_func(*args, **kwargs):\n'
        '    raise RuntimeError("flash_attn real no está instalado. Usa attn_implementation=\'eager\' para CPU.")\n',
        encoding="utf-8"
    )

    (fa_dir / "bert_padding.py").write_text(
        '# flash_attn.bert_padding stub\n\n'
        'def index_first_axis(*args, **kwargs):\n'
        '    raise RuntimeError("flash_attn real no está instalado.")\n\n'
        'def pad_input(*args, **kwargs):\n'
        '    raise RuntimeError("flash_attn real no está instalado.")\n\n'
        'def unpad_input(*args, **kwargs):\n'
        '    raise RuntimeError("flash_attn real no está instalado.")\n',
        encoding="utf-8"
    )

    print(f"✓ Stub de flash_attn instalado en: {fa_dir}")
    print("  Ya puedes ejecutar generate_captions.py sin errores de flash_attn.")


if __name__ == "__main__":
    install_stub()