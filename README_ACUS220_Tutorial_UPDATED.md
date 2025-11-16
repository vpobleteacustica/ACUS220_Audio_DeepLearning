<p align="left">
  <img src="escudo/uach_logo.png" alt="Universidad Austral de Chile" width="120" style="vertical-align:middle; margin-right:12px;">
</p>

# ACUS220 - Acústica Computacional con Python
## Instituto de Acústica - Universidad Austral de Chile
### Clasificando muestras de audio con deep learning (demo mínima)

Bienvenido/a. Esta guía te lleva paso a paso desde clonar el repositorio hasta entrenar un modelo CNN simple para clasificar señales acústicas.

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

## 2) Crear y activar entorno Conda
```bash
conda env create -f environment.yml
conda activate audio_deeplearning
```

## 3) Verificar el entorno
```bash
python scripts/100_check_env.py
```

> Si aparece algún error de importación, instala manualmente:
> ```bash
> pip install librosa soundfile torch torchvision torchaudio matplotlib pandas scikit-learn
> ```

Debe imprimir las versiones de librerías y **Shapes -> STFT/Log‑Mel/MFCC**. Si ves “**Entorno OK.**”, vamos bien.

---

## 4) PARTE A – Pipeline rápido (sin splits, demo de la clase)

Esta parte usa todo tiny_dataset junto (sin separar en train/val/test). Es ideal para explicar conceptos básicos de:
	•	extracción de features,
	•	visualización de espectrogramas,
	•	entrenamiento,
	•	seeds,
	•	e inferencia.

### 4A) Construir el dataset de features (sin splits)

Genera archivos `.npz` + `metadata.csv` desde `data/raw/tiny_dataset/`:

```bash
python scripts/200_build_tiny_dataset.py
```

Esto crea:

```
data/processed/tiny_specs/
├── <clase>/*.npz
└── metadata.csv
```

## Nota: Sobre los features extraídos:

La siguiente nota es válida tanto para `200_build_tiny_dataset.py`, como también para `210_split_tiny_dataset.py`.

El script: `200_build_tiny_dataset.py`
	•	Toma audios desde data/raw/tiny_dataset/
	•	Genera todas las features:
	•	STFT dB
	•	Log-Mel dB
	•	MFCC
	•	Gammatone (opcional)
	•	Guarda todo en los .npz
	•	Pero la CNN sólo usa mel_db, que corresponde al Log-Mel spectrogram.

El script: `210_split_tiny_dataset.py`
	•	Lee los audios, divide en train/val/test
	•	Genera exactamente los mismos features por cada WAV, incluyendo:
	•	STFT dB
	•	Log-Mel dB
	•	MFCC
	•	Gammatone (si existe)
	•	Y nuevamente, la CNN TinyMelCNN usa solamente mel_db (Log-Mel) al entrenar.



¿Por qué el modelo CNN que entrenaremos utiliza sólo el feature Log-Mel? La respuesta breve es que:
	•	Es la representación más estable para redes convolucionales,
	•	Mantiene buena resolución temporal y espectral,
	•	Es estándar en deep learning aplicado a audio.

Conclusión: Aunque podemos generar varias representaciones, la CNN TinyMelCNN entrena únicamente con el Log-Mel spectrogram. También, se podría hacer que:
	•	El modelo entrene con `MFCC`,
	•	o con `STFT`,
	•	o con `Gammatone`,
	•	o incluso concatenar features (`multi-branch CNN`).

Deberías ver un resumen de conteo por clase y el archivo:
```
data/processed/tiny_specs/metadata.csv
```

## 5A) Visualizar ejemplos de espectrogramas Log-Mel

Esta figura es muy valiosa de analizar. La figura presenta ejemplos de espectrogramas Log-Mel (mel_db) para cada una de las cinco especies del tiny_dataset. Cada bloque corresponde a una clase distinta, y cada columna es un ejemplo diferente dentro de esa clase.

¿Por qué es tan importante esta visualización? Porque aquí podemos ver lo mismo que verá la CNN cuando entrene…
y también lo que no ve.

1. Lo que la CNN sí ve: Son patrones visuales.

La CNN interpreta cada espectrograma como si fuera una imagen.

Detecta cosas como:
	•	Formas
	•	Bordes
	•	Texturas
	•	Regiones de energía concentrada
	•	Cambios bruscos o transiciones suaves
	•	Patrones repetitivos
	•	Manchas, líneas, franjas, pulsos, parches energéticos.

Es decir, la CNN observa morfología energética, no “sonido”.

En la figura puedes notar:
	•	Batrachyla leptopus → patrones difusos, distribuidos en banda media
	•	Batrachyla taeniata → franja más estable en banda media-baja
	•	Calyptocephalella gayi → energía más baja, difusa
	•	Pleurodema thaul → pulsos verticales muy marcados
	•	Porzana spiloptera → estructuras más ruidosas y amplias

Cada especie tiene una “huella visual” distinta.

2. Lo que la CNN no sabe: Aunque tú veas “frecuencias en Hz o kHz”, la CNN no tiene idea de eso.

La red no conoce:
	•	qué es un Hertz (Hz)
	•	qué es un 1 kHz
	•	qué parte del espectrograma es “agudo” o “grave”
	•	qué significa “frecuencia fundamental”, “armónicos” o “formantes”
	•	qué especie produce el sonido
	•	qué objeto físiológico (trayecto vocal de la especie) generó la onda.

Para la CNN, el eje vertical no son kHz: es simplemente la coordenada Y de una imagen.

3. Conclusión:

• Un espectrograma Log-Mel transforma un sonido en una imagen de energía.
• La CNN aprende a reconocer patrones visuales en esa imagen, no conceptos acústicos como frecuencia, kHz, resonancia o timbre.
• Lo que aprende es la morfología del sonido, tal como un ojo entrenado reconoce formas.

Por eso esta figura es tan valiosa: muestra claramente las “formas acústicas” que cada especie deja en su espectrograma, y revela por qué una CNN es capaz de clasificarlas aun sin saber nada de acústica.

```bash
python scripts/250_preview_tiny.py \
    --metadata data/processed/tiny_specs/metadata.csv \
    --feature mel_db \
    --n-per-class 5 \
    --out figures/tiny_specs_preview.png
```

Salida esperada:
```
data/figures/tiny_specs_preview.png
```
Debieras ver una imagen de 5 filas = 5 especies, por 5 columnas = 5 ejemplos de cada especie.


## 6A) Pipeline simple en un solo paso (opcional)

Ejecuta:
```bash
./scripts/2500_run_tiny_pipeline.sh
```

Este bash hace, en orden:
	1.	Construcción del dataset (200_build_tiny_dataset.py)
	2.	Visualización de ejemplos (250_preview_tiny.py)
	3.	Entrenamiento simple con Log-Mel (300_train_tiny_cnn.py)

## 7A) Entrenamiento simple (CNN pequeña con Log-Mel)

¿Qué significan epochs, batch-size y learning rate? 

• Épocas (epochs)

Una época es una pasada completa por todo el dataset de entrenamiento.
	•	--epochs 25 significa que el modelo verá 25 veces todos los ejemplos.
	•	Más épocas → más aprendizaje (pero también más riesgo de sobreajuste).
	•	Menos épocas → entrenamiento más rápido pero tal vez insuficiente.

Una metáfora: Si estudias tu cuaderno completo una vez = 1 época.

• Tamaño de batch (batch-size)

El batch-size significa cuántos ejemplos procesa la red al mismo tiempo antes de actualizar los pesos.
	•	--batch-size 8 significa que la red mira 8 espectrogramas por vez, calcula el error, y luego ajusta los pesos.
	•	Batch pequeño → aprendizaje más ruidoso pero más estable.
	•	Batch grande → aprendizaje más suave pero requiere más memoria.

Una metáfora: Estudias de a 8 ejercicios antes de revisar cómo vas.

• El learning rate controla qué tan grande es el paso de aprendizaje en cada actualización.
	•	--lr 1e-3 significa un paso pequeño pero seguro.
	•	Learning rate muy grande → aprendizaje inestable.
	•	Learning rate muy pequeño → la red aprende muy lento.

Una metáfora: Si caminas hacia una meta con pasos muy grandes, puedes pasarte. En cambio, es más seguro llegar si tus pasos son pequeños.

# Resumen breve

| Parámetro | Qué controla | Ejemplo | Explicación|
|---------|-------------|---------|-------------|
| **epochs** | cuántas veces se recicla el dataset | 25 | estudiar el cuaderno 25 veces |
| **batch-size** | cuántos ejemplos se ven por paso| 8 | estudiar de a 8 ejercicios antes de corregir|
| **lr** | tamaño del paso de aprendizaje | 1e-3| pasos pequeños, estables |


Ejecuta entrenamiento con una o varias semillas (recomendado para reproducibilidad):
```bash
python -m scripts.300_train_tiny_cnn --epochs 25 --batch-size 8 --lr 1e-3 --seeds 42,7,123
```
---

Por cada seed se crea:
```
data/processed/tiny_specs/cnn_run/seed_<SEED>/
├── best_model.pt
├── history.csv              # pérdida y accuracy por época
├── curvas_loss_acc.png      # gráfico loss/accuracy train-val
├── confusion_matrix.png     # matriz de confusión (demo simple)
└── class_names.json
```

Resumen general: 
```
data/processed/tiny_specs/cnn_run/run_summary.csv
```

Contiene, por seed: mejor época, mejor val_acc, etc.

---

# ¿Qué significa control de aleatoriedad y uso de *semillas* (seed) en Deep Learning?

En modelos de deep learning, incluso cuando usamos exactamente el mismo script, los resultados pueden variar entre ejecuciones porque existen múltiples fuentes internas de aleatoriedad.

---

## 1. Inicialización aleatoria de pesos
Cada capa de la red neuronal inicia sus pesos con valores aleatorios.  
Una seed fija ese punto de partida.

---

## 2. Barajar el dataset
El `DataLoader` mezcla aleatoriamente los ejemplos antes de cada época.  
Cambiar la seed cambia el orden → cambia el gradiente → cambia el entrenamiento.

---

## 3. Augmentación aleatoria
Rotaciones, ruidos, shifts, etc., son aplicados con azar.  
Sin seed, dos entrenamientos nunca verán las mismas imágenes.

---

# 4 ¿Qué es entonces una *semilla* (seed)?

Una *seed* es un número entero que **fija** todos los generadores de números aleatorios:

```python
import torch, numpy as np, random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Si usamos la misma seed → obtendremos exactamente el mismo resultado.

---

# 5. ¿Por qué entrenar con **múltiples** seeds?

Porque un único entrenamiento puede ser afortunado… o no?.

Entrenar con diferentes semillas permite:

- Evaluar la **estabilidad** del modelo  
- Identificar si un resultado fue **casual**  
- Obtener métricas más **robustas** (media ± desviación estándar)  
- Comparar arquitecturas de forma **justa**.  

Ejemplo:

```bash
python -m scripts.300_train_tiny_cnn --epochs 25 --batch-size 8 --lr 1e-3 --seeds 42,7,123
```


# 6. Metáfora didáctica.

Entrenar una red neuronal es como plantar un árbol: Si cambiamos la semilla biológica, el árbol será similar… pero nunca idéntico.

Varias seeds = varios árboles → puedes comparar cuál creció mejor.

# Resumen general

| Concepto | Explicación |
|---------|-------------|
| **Seed** | Número que controla la aleatoriedad |
| **Reproducibilidad** | Misma seed → mismo resultado |
| **Variabilidad natural** | Distintas seeds → distintas curvas de entrenamiento |
| **Buena práctica** | Reportar media ± desviación estándar |

---

## 8A) Inferencia: clasificar un WAV

Usa el mejor modelo del resumen (por defecto se elige el de **máxima accuracy**):
```bash
python -m scripts.350_infer_one --wav data/raw/tiny_dataset/Batrachyla_taeniata/ejemplo.wav
```
selecciona tú el ejemplo.wav que quieras desde tus carpetas de audio por especie.

Salida esperada:
```
WAV: ...
Modelo: data/processed/tiny_specs/cnn_run/seed_123/best_model.pt
Predicción: Batrachyla_taeniata (p=0.563)
Vista: data/processed/tiny_specs/cnn_run/seed_123/infer_*.png
```

> **Tip**: Para inferir por semilla específica: `--seed 42`


## 9A) Inferencia por lote (batch)

Podemos clasificar todos los WAV de una carpeta y ver la distribución de predicciones:

```bash
python scripts/360_infer_batch.py \
    --wav-dir data/raw/tiny_dataset/Batrachyla_leptopus
```

Esto produce, para cada clase / carpeta que elijas:

```
data/processed/tiny_specs/cnn_run/seed_<SEED>/
├── inference_results_<NOMBRE_CLASE>.csv    # predicciones por archivo
├── summary_<NOMBRE_CLASE>.png              # gráfico de barras (clases predichas)
└── inference_previews_<NOMBRE_CLASE>/*.png # espectrogramas + etiqueta modelo
```

---

## 1B) PARTE B – Pipeline con train / val / test (configuración “seria”)

En esta parte separamos explícitamente los datos para:
	•	entrenar (train),
	•	ajustar hiperparámetros (val),
	•	medir desempeño final (test).

Todo se hace de forma reproducible usando seeds.

### 1. ¿Por qué separar en train, val y test?

En aprendizaje profundo, nunca se entrena y evalúa un modelo con los mismos datos. Por eso dividimos el dataset en tres partes con roles muy distintos:

1. Train (entrenamiento)

Es el conjunto que la red usa directamente para aprender. Aquí el modelo ajusta sus parámetros internos viendo miles de ejemplos.

Una metáfora: Imagina que eres un deportista de alto rendimiento, **train** sería tu rutina de ejercicios que haces durante una práctica.

2. Validation (val)

Usamos este split para **medir el rendimiento** durante el entrenamiento, sin afectar al modelo. Nos sirve para:
	•	elegir hiperparámetros (lr, batch-size, n_mels, etc.),
	•	decidir cuántas épocas entrenar,
	•	seleccionar el mejor modelo (early stopping).

Tu metáfora: es el ensayo general que harías antes de tu competencia; no cuenta para el resultado deportivo que se trate, pero te indica cómo vas.

🔹 3. Test (evaluación final)

Este conjunto se usa una sola vez, al final. Nos mide el desempeño real del modelo en datos nunca antes vistos.

Tu metáfora: es la prueba oficial en tu competencia. Entras a la cancha y ahí realmente sabes qué tan estás, o en términos de un modelo de aprendizaje, qué tan bien **generaliza** el modelo.

# Resumen general

| Split | ¿Qué es? | ¿Para qué sirve?|
|---------|-------------|-------------|
| **train** | datos usados para aprender | ajustar pesos del modelo |
| **val**   | datos para validar durante el entrenamiento | ajustar hiperparámetros y elegir el mejor modelo |
| **test** | datos NO usados en el entrenamiento | medir el rendimiento real, final y honesto |


## 2B) Crear splits del tiny_dataset (train/val/test)

A partir de data/raw/tiny_dataset/<clase>/*.wav:

Ejecuta:
```bash
python scripts/210_split_tiny_dataset.py`
```

Esto crea:

```
data/raw/tiny_dataset_split/
├── train/<clase>/*.wav
├── val/<clase>/*.wav
└── test/<clase>/*.wav

data/raw/tiny_dataset_split/split_metadata.csv
```

El script va a imprimir cuántos archivos quedan en cada split y clase.

## 3B) Construir features por split

Ahora generamos .npz separados para cada split:

```bash
python scripts/220_build_tiny_dataset_from_split.py`
```

Salida:
```
data/processed/tiny_specs_split/
├── train/<clase>/*.npz
├── val/<clase>/*.npz
├── test/<clase>/*.npz
└── metadata_split.csv   # columnas: wav, npz, label, split
```

## 4B) Entrenamiento con splits (TinyMelCNN)

Entrenamos con **train**, validamos con **val** y evaluamos con **test**:

```bash
python scripts/300_train_tiny_cnn_split.py \
    --epochs 20 \
    --batch-size 16 \
    --seeds 42,7,123
```

Salida por seed:

```
data/processed/tiny_specs_split/cnn_run_split/seed_<SEED>/
├── best_model.pt
├── history.csv
├── curvas_loss_acc.png          # loss/accuracy train vs val
├── confusion_matrix_counts.csv  # matriz de confusión (conteos)
├── confusion_matrix_norm.csv    # matriz de confusión (proporciones por fila)
├── confusion_matrix.png         # figura con valores normalizados (0–1)
├── test_metrics.json            # métricas globales de test
└── class_names.json
```

Resumen global de todas las seeds:

```bash
data/processed/tiny_specs_split/cnn_run_split/run_summary_split.csv
```

con columnas como: seed, best_epoch, best_val_acc, test_loss, test_acc, best_model.

*Comentario didáctico*:
La **matriz de confusión normalizada** (confusion_matrix_norm.csv + confusion_matrix.png) muestra, por cada clase real/verdadera, la distribución de probabilidades de predicción.
Cada fila suma 1.0 → se interpreta como:

“Dado que la clase real es X, ¿con qué probabilidad el modelo predice Y?”

## C) PARTE C: Próximos pasos futuros
- Mapas de activación
- Gammatone filters
- Más datos
- Modelos preentrenados (VGGish, YAMNet)
