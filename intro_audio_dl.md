<p align="left">
  <img src="escudo/uach_logo.png" alt="Universidad Austral de Chile" width="120" style="vertical-align:middle; margin-right:12px;">
</p>

# ACUS220 – Acústica Computacional con Python - Instituto de Acústica, Universidad Austral de Chile  
### Clasificando muestras de audio & deep learning (demo mínima)

---

## 1. ¿Qué es un espectrograma?

Un **espectrograma** es una representación visual de cómo cambia la energía de las frecuencias de una señal de audio a lo largo del tiempo.  
- En el eje **x** se muestra el tiempo.  
- En el eje **y** se muestran las **frecuencias** (Hz) en escala lineal.  
- El color indica la **intensidad (amplitud o potencia)** en decibeles (dB) en escala logarítmica.  

Se calcula aplicando la **Transformada Rápida de Fourier (STFT)** en ventanas de tiempo corto.  
Cada ventana revela qué frecuencias están presentes en ese instante.  
Esto permite visualizar patrones acústicos presentes en la señal de audio como repertorios distintos de cantos, vocalizaciones, o ruidos.

---

## 2. ¿Por qué usamos la escala Log-Mel?

El **espectrograma Log-Mel** transforma la escala lineal de frecuencias a una escala perceptual llamada **escala Mel**, que aproxima cómo el oído humano percibe el sonido.  

- Las frecuencias bajas se representan con más detalle (mayor resolución).  
- Las altas frecuencias se comprimen, imitando la sensibilidad auditiva.  
- Luego se aplica una **escala logarítmica** a la energía (en dB), que se alinea con la percepción de intensidad del oído humano.

El resultado es un **mapa de energía auditivamente relevante**, ideal para alimentar modelos de aprendizaje profundo, especialmente CNNs.

---

## 3. ¿Qué son los *audio features*?

Los **features acústicos** son valores numéricos que resumen información relevante del sonido.  
Ejemplos comunes:
- Energía total o RMS (amplitud promedio).  
- Centroides espectrales (frecuencias “promedio”).  
- Bandwidth, roll-off, zero-crossing rate.  
- MFCCs, Log-Mel, Chroma, etc.

Estos valores condensan la información de los espectrogramas en una forma que las redes neuronales pueden analizar más fácilmente.

---

## 4. ¿Qué son los MFCC (Mel-Frequency Cepstral Coefficients)?

Los coeficientes **MFCCs** son un conjunto de valores numéricos que representan la **envolvente espectral** del sonido.  
Se obtienen a partir del espectrograma Log-Mel, aplicando una transformada coseno discreta (DCT) que concentra la información más relevante en pocos valores (normalmente se usan entre 13 a 20 coeficientes).  

Son ampliamente usados en:
- Reconocimiento de voz.  
- Clasificación de especies animales.  
- Detección de sonidos urbanos.  

En resumen, los MFCC resumen “cómo suena” una señal de forma compacta y perceptualmente significativa.

---

## 5. ¿Qué es el *Deep Learning* aplicado al audio?

El **Deep Learning** permite aprender representaciones jerárquicas a partir de datos.  
En audio, esto significa que el modelo puede descubrir automáticamente:
- Bordes espectrales (frecuencias dominantes).  
- Formas de energía (modos de vibración).  
- Patrones temporales o espaciales.  

Las **redes neuronales convolucionales (CNNs)** son particularmente adecuadas para analizar espectrogramas, ya que funcionan de manera análoga a la visión por computador: aprenden filtros locales que detectan patrones.

---

## 6. Arquitectura CNN usada en esta demo

La red usada en el demo (`TinyMelCNN`) es una **CNN pequeña pero representativa**.  
Consta de tres bloques convolucionales seguidos por capas densas:

```
Input (1 × 64 × 173)  # canal único, espectrograma Log-Mel
↓
Conv2D(1→16) + ReLU + MaxPool2D
↓
Conv2D(16→32) + ReLU + MaxPool2D
↓
Conv2D(32→64) + ReLU + AdaptiveAvgPool2D(4×4)
↓
Flatten → Linear(64*4*4 → 128) + ReLU + Dropout
↓
Linear(128 → n_classes)
```

- **Conv2D** aprende filtros que detectan patrones frecuenciales o temporales.  
- **ReLU** introduce no linealidad.  
- **MaxPool** reduce dimensionalidad y destaca las características más importantes.  
- **AdaptiveAvgPool** ajusta el tamaño de salida independientemente del tamaño de entrada.  
- **Linear** y **Softmax** generan las probabilidades de clase final.

Aunque simple, este modelo es suficiente para observar el flujo completo: *desde el audio hasta la clasificación*.

---

## 7. Próximos pasos a futuro:

1. Explorar los **mapas de activación** de la CNN.  
2. Introducir **Gammatone filters** como alternativa perceptiva a Mel.  
3. Experimentar con **más clases y más datos**.  
4. Comparar CNNs pequeñas con modelos preentrenados (VGGish, YAMNet).  

---

Material docente para el curso ACUS220 - Acústica Computacional con Python (Instituto de Acústica - Universidad Austral de Chile UACh). Uso educativo.  
*Contacto: Víctor Poblete — vpoblete@uach.cl*.
