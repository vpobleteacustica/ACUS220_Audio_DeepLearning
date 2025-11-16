<p align="left">
  <img src="escudo/uach_logo.png" alt="Universidad Austral de Chile" width="120" style="vertical-align:middle; margin-right:12px;">
</p>

# README.md — versión actualizada para la Clase 2

ACUS220 – Acústica Computacional con Python
Instituto de Acústica — Universidad Austral de Chile
Mini-demo: clasificación de audio usando espectrogramas Log-Mel + CNN pequeña

## Flujo general del proyecto
	1.	Chequear entorno
	2.	Construir mini-dataset de features (WAV → NPZ con Log-Mel/MFCC/STFT)
	3.	(Clase 2) Particionar dataset en train / val / test
	4.	Entrenar CNN pequeña (con seeds múltiples)
	5.	Ver curvas, matriz de confusión y probabilidades
	6.	Inferencia con un solo WAV
	7.	Pipeline automático (opcional)

## Instalación del entorno

Recomendado: Conda.

```bash
conda env create -f environment.yml
conda activate audio_deeplearning
```
Verificar
```bash
python scripts/100_check_env.py
```

Debe mostrar versiones de librerías + shapes de STFT / Log-Mel / MFCC.
Si falta algo:
```bash
pip install librosa soundfile torch torchvision torchaudio matplotlib pandas scikit-learn
```

## 1) Construir dataset de features (versión simple, sin splits)

Esto genera:

```
data/processed/tiny_specs/
    <clase>/<archivo>.npz
    metadata.csv
```

Comando:
```bash
python scripts/200_build_tiny_dataset.py
```

¿Qué features se extraen aquí?

El archivo .npz guarda:
	•	Log-Mel (principal para la CNN)
	•	STFT en dB
	•	MFCC
	•	Opcional: Gammatone (si está disponible)

Pero la CNN solo usa Log-Mel dB, porque es muy estable, robusto y funciona como una “imagen” donde la CNN ve **formas, bordes, patrones y energía, no Hz reales**.

## 2) Visualizar ejemplos

```bash
python scripts/250_preview_tiny.py
```
Genera un panel por clase con espectrogramas Log-Mel.

## 3) Construir dataset con splits (train/val/test)

### (A) División estratificada

```bash
python scripts/210_split_tiny_dataset.py
```

Genera:
```
data/raw/tiny_dataset_split/
    train/<clase>/*.wav
    val/<clase>/*.wav
    test/<clase>/*.wav
```

### (B) Construir features para cada split
```bash
python scripts/220_build_tiny_dataset_from_split.py
```

Salida:
```
data/processed/tiny_specs_split/
    train/<clase>/*.npz
    val/<clase>/*.npz
    test/<clase>/*.npz
    metadata_split.csv
```

## 4) Entrenamiento de la CNN (multi-seed)
```bash
python scripts/300_train_tiny_cnn_split.py \
    --epochs 20 \
    --batch-size 16 \
    --seeds 42,7,123
```

Cada carpeta incluye:
	•	best_model.pt
	•	history.csv
	•	curvas_loss_acc.png
	•	confusion_matrix.png (normalizada + conteos)
	•	confusion_matrix_norm.csv
	•	test_metrics.json

Y un run_summary_split.csv consolidado.

## 5) Inferencia — clasificar un WAV
```bash
python scripts/350_infer_one.py \
    --wav data/raw/tiny_dataset/Batrachyla_taeniata/audio601_label28.wav
```

## 6) Pipeline completo (opcional)

Comandos:
```bash
chmod +x scripts/2500_run_tiny_pipeline.sh
```
Y luego,
```bash
bash scripts/2500_run_tiny_pipeline.sh
```

## Documentación extendida

Revisa el tutorial completo en:

TUTORIAL.md

Es el documento guía para la clase, paso a paso, con explicaciones pedagógicas.)

## Agradecimientos

Este repositorio fue desarrollado para la asignatura ACUS220 - Acústica Computacional con Python,
Instituto de Acústica, Universidad Austral de Chile.