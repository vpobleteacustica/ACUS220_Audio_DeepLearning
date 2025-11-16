#!/usr/bin/env bash
# -------------------------------------------------------------
# 2500_run_tiny_pipeline.sh
#
# Construye el tiny-dataset (NPZ) y genera la figura de preview
# para la clase ACUS220.
#
# Uso:
#   bash scripts/2500_run_tiny_pipeline.sh
#
# Asegúrate de tener activado tu entorno:
#   conda activate audio_deeplearning
# -------------------------------------------------------------

set -e  # Abort on error

echo "🐍 Ejecutando pipeline Tiny Dataset para ACUS220..."
echo "---------------------------------------------------"

# 1. Construir tiny dataset
echo "Paso 1/2: Construyendo tiny dataset..."
python scripts/200_build_tiny_dataset.py

# 2. Generar figure preview
echo "Paso 2/2: Generando figura de preview..."
python scripts/250_preview_tiny.py \
    --metadata data/processed/tiny_specs/metadata.csv \
    --feature mel_db \
    --n-per-class 5 \
    --out figures/tiny_specs_preview.png

echo "---------------------------------------------------"
echo "Pipeline completo."
echo "Figura generada en: figures/tiny_specs_preview.png"