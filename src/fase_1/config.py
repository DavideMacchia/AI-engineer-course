"""Costanti del progetto: costi di business, seme, percorsi, split.

Un solo posto da toccare per cambiare i parametri dell'esperimento.
"""
from pathlib import Path

# Costi di business: da qui si ricava la soglia, non dal modello
COSTO_FP = 170   # sconto offerto a chi sarebbe rimasto comunque
COSTO_FN = 400   # cliente perso senza aver fatto niente

SEME = 42        # senza, accuratezza e AUC ballano di ±0.01 a ogni run

# Percorsi calcolati dalla posizione di questo file, non dalla cartella di lavoro.
# config.py sta in src/fase_1/ -> tre livelli sopra c'è la radice del progetto.
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
PERCORSO_DATI = CARTELLA_PROGETTO / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Split: Train 70% / Validation 15% / Test 15% (stratificato)
QUOTA_TEMP = 0.3   # val + test insieme
QUOTA_TEST = 0.5   # metà di temp -> 15% del totale

PERCORSO_MLFLOW = f"sqlite:///{CARTELLA_PROGETTO / 'mlflow.db'}"