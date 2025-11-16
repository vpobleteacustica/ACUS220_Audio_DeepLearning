# Control de Aleatoriedad y Uso de *Semillas* en Deep Learning

En modelos de deep learning, incluso cuando usamos exactamente el mismo código, los resultados pueden variar entre ejecuciones porque existen múltiples fuentes internas de aleatoriedad.

---

## 1. Inicialización aleatoria de pesos
Cada capa de la red neuronal inicia sus pesos con valores aleatorios.  
Una seed fija ese punto de partida.

---

## 2. Barajado del dataset
El `DataLoader` mezcla aleatoriamente los ejemplos antes de cada época.  
Cambiar la seed cambia el orden → cambia el gradiente → cambia el entrenamiento.

---

## 3. Augmentación aleatoria
Rotaciones, ruidos, shifts, etc., son aplicados con azar.  
Sin seed, dos entrenamientos nunca verán las mismas imágenes.

---

# ¿Qué es una *semilla* (seed)?

Una *seed* es un número entero que fija todos los generadores de números aleatorios:

```python
import torch, numpy as np, random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Si usamos la misma seed → obtenemos exactamente el mismo resultado.

---

# ¿Por qué entrenar con **múltiples** seeds?

Porque un único entrenamiento puede ser afortunado… o no.

Entrenar con diferentes semillas permite:

- Evaluar la **estabilidad** del modelo  
- Identificar si un resultado fue casual  
- Obtener métricas más robustas (media ± desviación estándar)  
- Comparar arquitecturas de forma justa  

Ejemplo:

```bash
python -m scripts.300_train_tiny_cnn --epochs 25 --batch-size 8 --lr 1e-3 --seeds 42,7,123
```

---

# Metáfora didáctica (útil para clase)

Entrenar una red neuronal es como plantar un árbol:  
Si cambias la semilla biológica, el árbol será similar… pero nunca idéntico.

Varias seeds = varios árboles → puedes comparar cuál creció mejor.

---

# Resumen general

| Concepto | Explicación |
|---------|-------------|
| **Seed** | Número que controla la aleatoriedad |
| **Reproducibilidad** | Misma seed → mismo resultado |
| **Variabilidad natural** | Distintas seeds → distintas curvas de entrenamiento |
| **Buena práctica** | Reportar media ± desviación estándar |

---

Este archivo está listo para usarlo como parte de tu material docente en ACUS220 🎓.
