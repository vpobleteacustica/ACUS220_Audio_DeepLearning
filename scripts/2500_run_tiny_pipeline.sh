#!/usr/bin/env bash
# -------------------------------------------------------------
# 2500_run_tiny_pipeline.sh
#
# Pipeline mínimo para ACUS220:
#   1) Chequear entorno
#   2) Construir tiny_dataset (NPZ sin splits)
#   3) Generar figura de preview (mel_db)
#   4) Crear splits train/val/test
#   5) Construir features NPZ por split
#   6) Entrenar CNN pequeña (multi-seed) sobre train/val y evaluar en test
#
# Uso:
#   bash scripts/2500_run_tiny_pipeline.sh
#
# Asegúrate de tener activado tu entorno:
#   conda activate audio_deeplearning
# -------------------------------------------------------------

set -e  # Abort on error ante cualquier fallo

echo "🐍 Ejecutando pipeline Tiny Dataset para ACUS220..."
echo "---------------------------------------------------"

# 0. Chequear entorno
echo "Paso 0/5: Chequeando entorno..."
python scripts/100_check_env.py

# 1. Construir tiny dataset (sin splits)
echo "Paso 1/5: Construyendo tiny_dataset (NPZ sin splits)..."
python scripts/200_build_tiny_dataset.py

# 2. Generar figura de preview (mel_db)
echo "Paso 2/5: Generando figura de preview (mel_db)..."
python scripts/250_preview_tiny.py \
    --metadata data/processed/tiny_specs/metadata.csv \
    --feature mel_db \
    --n-per-class 5 \
    --out figures/tiny_specs_preview.png

# 3. Crear splits train/val/test a partir de los WAV
echo "Paso 3/5: Creando splits train/val/test..."
python scripts/210_split_tiny_dataset.py

# 4. Construir features NPZ para cada split
echo "Paso 4/5: Construyendo features (NPZ) por split..."
python scripts/220_build_tiny_dataset_from_split.py

# 5. Entrenar CNN pequeña (multi-seed) usando los splits
echo "Paso 5/5: Entrenando CNN pequeña (multi-seed) con train/val/test..."
python scripts/300_train_tiny_cnn_split.py \
    --epochs 20 \
    --batch-size 16 \
    --seeds 42,7,123

echo "---------------------------------------------------"
echo "✅ Pipeline completo."
echo "Figura preview:      figures/tiny_specs_preview.png"
echo "Resultados de CNN:   data/processed/tiny_specs_split/cnn_run_split/"
echo "  ↳ Ver history.csv, curvas_loss_acc.png y confusion_matrix.png por seed."