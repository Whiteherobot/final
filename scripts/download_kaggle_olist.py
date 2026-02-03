"""
Descarga el dataset Olist desde Kaggle a data/olist
Requisitos:
- Instalar Kaggle CLI: pip install kaggle
- Configurar credenciales Kaggle (kaggle.json)
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "olist"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "olistbr/brazilian-ecommerce"

print("Descargando dataset:", DATASET)
print("Destino:", DATA_DIR)

cmd = [
    "kaggle",
    "datasets",
    "download",
    "-d",
    DATASET,
    "-p",
    str(DATA_DIR),
    "--unzip",
]

try:
    subprocess.check_call(cmd)
    print("Descarga completada.")
except FileNotFoundError:
    print("Kaggle CLI no está instalado. Ejecuta: pip install kaggle")
    raise
except subprocess.CalledProcessError as exc:
    print("Error al descargar el dataset.")
    raise exc
