# ACUS220 — Clasificando muestras de audio & deep learning (demo mínima)

## Flujo básico
1) Chequear entorno  
2) Construir mini-dataset (WAV → NPZ con Log-Mel/MFCC)  
3) Entrenar CNN pequeñita  
4) Ver curvas y matriz de confusión

## Comandos rápidos
```bash
python scripts/100_check_env.py
python scripts/200_build_tiny_dataset.py
python scripts/300_train_tiny_cnn.py
