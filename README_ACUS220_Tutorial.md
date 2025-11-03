<p align="left">
  <img src="escudo/uach_logo.png" alt="Universidad Austral de Chile" width="120" style="vertical-align:middle; margin-right:12px;">
</p>

# ACUS220 – Acústica Computacional con Python – Instituto de Acústica, Universidad Austral de Chile
### Clasificando muestras de audio & deep learning (demo mínima)

Bienvenido/a... Esta guía te lleva **paso a paso** desde clonar el repositorio hasta entrenar un clasificador sencillo con espectrogramas Log‑Mel y features Mel Frequency Cepstral Coefficients (MFCC) típicamente usados en audio.

> **Requisitos previos**: Git, Python 3.10+, y una de estas opciones para entornos: **Conda (recomendado)** o **venv + pip**.

---

## 1) Clonar el repositorio

```bash
git clone https://github.com/vpobleteacustica/ACUS220_Audio_DeepLearning.git
cd ACUS220_Audio_DeepLearning
```

Estructura mínima esperada:

```
ACUS220_Audio_DeepLearning/
├── data/
│   ├── raw/                # audios de entrada (tiny_dataset)
│   └── processed/          # se genera automáticamente
├── notebooks/              # cuadernos Jupyter opcionales
├── scripts/                # scripts ejecutables (chequeo, dataset, train, infer)
├── src/                    # utilidades y funciones de audio/features
├── environment.yml         # entorno Conda (recomendado)
├── requirements.txt        # alternativa con pip
└── README.md
```

---

## 2) Crear y **activar** el entorno de trabajo

### Opción A — Usando Conda (recomendada)
1) Crear el entorno con `environment.yml`:
```bash
conda env create -f environment.yml
```

2) Activar el entorno:
```bash
conda activate audio_deeplearning
```

3) (Si ya existía) Actualizar/limpiar dependencias:
```bash
conda env update -f environment.yml --prune
```

4) Verificar el entorno:
```bash
python scripts/100_check_env.py
```
Debe imprimir las versiones de librerías y **Shapes -> STFT/Log‑Mel/MFCC**. Si ves “**Entorno OK.**”, vamos bien.

---

### Opción B — Usando `requirements.txt` (pip + venv)

1) Crear un entorno virtual:
```bash
python -m venv audio_dl_env
# Mac/Linux
source audio_dl_env/bin/activate
# Windows (PowerShell)
audio_dl_env\Scripts\Activate.ps1
```

2) Instalar dependencias:
```bash
pip install -r requirements.txt
```

3) Verificar el entorno:
```bash
python scripts/100_check_env.py
```

> Si aparece algún error de importación, instala manualmente:
> ```bash
> pip install librosa soundfile torch torchvision torchaudio matplotlib pandas scikit-learn
> ```

---

## 3) Preparar un **mini‑dataset** (tiny dataset)

Coloca tus archivos de audio en formato WAV en subcarpetas por clase dentro de `data/raw/tiny_dataset/`, por ejemplo:

```
data/raw/tiny_dataset/
├── Batrachyla_leptopus/
├── Batrachyla_taeniata/
├── Calyptocephalella_gayi/
├── Pleurodema_thaul/
└── Porzana_spiloptera/
```

> **Sugerencia didáctica**: 10–20 WAV por clase (2 s, 22.05 kHz) es suficiente para el ejercicio.

Construir el mini‑dataset (genera `.npz` y `metadata.csv`):

```bash
python -m scripts.200_build_tiny_dataset
```

Deberías ver un resumen de conteo por clase y el archivo:
```
data/processed/tiny_specs/metadata.csv
```

**Previsualizar ejemplos** (grids de STFT/Log‑Mel/MFCC por clase):
```bash
python -m scripts.250_preview_tiny
```
Salida esperada:
```
data/processed/tiny_specs/_preview_*.png
```

---

## 4) Entrenamiento (CNN pequeña con Log‑Mel)

Ejecuta entrenamiento con una o varias semillas (recomendado para reproducibilidad):
```bash
python -m scripts.300_train_tiny_cnn --epochs 25 --batch-size 8 --lr 1e-3 --seeds 42,7,123
```

Artefactos por semilla:
```
data/processed/tiny_specs/cnn_run/seed_<SEED>/
├── best_model.pt
├── loss_curves.png
├── confusion_matrix.png
├── report.txt           # métricas de validación
└── test_report.txt      # métricas en test (si aplica)
```
Resumen general:
```
data/processed/tiny_specs/cnn_run/run_summary.csv
```

---

## 5) Inferencia: clasificar **un WAV**

Usa el mejor modelo del resumen (por defecto se elige el de **máxima accuracy**):
```bash
python -m scripts.350_infer_one --wav data/raw/tiny_dataset/Batrachyla_taeniata/ejemplo.wav
```
Salida esperada:
```
WAV: ...
Modelo: data/processed/tiny_specs/cnn_run/seed_123/best_model.pt
Predicción: Batrachyla_taeniata (p=0.563)
Vista: data/processed/tiny_specs/cnn_run/seed_123/infer_*.png
```

> **Tip**: Para inferir por semilla específica: `--seed 42`

---

## 6) (Opcional) Notebook guía

Si prefieres una experiencia paso a paso en Jupyter, abre el cuaderno:
```
notebooks/000_from_audio_to_cnn.ipynb
```
Asegúrate de seleccionar el **Kernel** del entorno creado (`audio_deeplearning` o `audio_dl_env`).

---

## 7) Solución de problemas (FAQ)

**Q1.** `ModuleNotFoundError: No module named 'src'`  
→ Ejecuta los scripts **desde la raíz del repo** y no desde subcarpetas. Verifica el *cwd* con `pwd` y usa `python -m scripts.<nombre>`.

**Q2.** `soundfile/libsndfile` error en macOS  
→ Instala libsndfile del sistema (Homebrew): `brew install libsndfile`.

**Q3.** `torch.cuda.is_available(): False`  
→ Es normal si no tienes GPU. El curso está pensado para CPU.

**Q4.** ¿Cuántos WAV mínimos por clase?  
→ Para el tutorial, 10–20 está bien. Para resultados más estables, usa >50 por clase.

**Q5.** ¿Cómo reproducir resultados?  
→ Usa `--seeds 42,7,123` y conserva la misma estructura de carpetas y versiones del entorno.

---

## 8) Estructura de carpetas (generada durante el flujo)

```
data/
├── raw/
│   └── tiny_dataset/<clase>/*.wav
└── processed/
    └── tiny_specs/
        ├── <clase>/*.npz
        ├── metadata.csv
        └── cnn_run/
            ├── seed_42/...
            ├── seed_7/...
            ├── seed_123/...
            └── run_summary.csv
```

---

## 9) Licencia y crédito

Material docente para el curso ACUS220 - Acústica Computacional con Python (Instituto de Acústica - Universidad Austral de Chile UACh). Uso educativo.  
Contacto: Víctor Poblete — *vpoblete@uach.cl*.
