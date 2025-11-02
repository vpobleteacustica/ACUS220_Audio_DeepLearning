#!/usr/bin/env bash
set -e

# 1) Carpetas
mkdir -p data/raw/tiny_dataset
mkdir -p data/processed
mkdir -p notebooks
mkdir -p scripts
mkdir -p src

# 2) .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc

# Data
data/processed/

# Notebooks
.ipynb_checkpoints/

# OS/IDE
.DS_Store
.vscode/
EOF

# 3) requirements.txt
cat > requirements.txt <<'EOF'
numpy
pandas
matplotlib
scikit-learn
librosa==0.10.2.post1
soundfile
torch
EOF

# 4) README.md
cat > README.md <<'EOF'
# ACUS220 — Audio & Deep Learning (demo mínima)

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