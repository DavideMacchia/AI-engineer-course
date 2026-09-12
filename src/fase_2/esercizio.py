"""Fase 2 — Stadio 0: dati e struttura di una rete neurale da zero (NumPy).

La traccia originale usa make_moons (2 feature -> architettura 2 -> 8 -> 8 -> 1).
Qui il dataset e' il Telco: l'input NON e' 2, e' il numero di colonne dopo il
preprocessing (one-hot sulle categoriche + scaling sulle numeriche) = 44.
Quindi l'architettura diventa  44 -> 8 -> 8 -> 1: cambia solo la dimensione
d'ingresso, i due hidden layer da 8 e l'output da 1 restano identici.

In questo stadio si prepara SOLO l'impalcatura: dati divisi e pesi inizializzati.
Il forward pass, la loss e la backpropagation arrivano negli stadi successivi.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEME = 42
QUOTA_TEST = 0.2  # 80% train / 20% test
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
PERCORSO_DATI = CARTELLA_PROGETTO / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Architettura: input calcolato dai dati, poi 8 -> 8 -> 1 come da traccia.
DIM_NASCOSTE = [8, 8]


# --------------------------------------------------------------------------- #
# 1. Dati: carico, pulisco, divido PRIMA di preprocessare
# --------------------------------------------------------------------------- #
def prepara_dati():
    """Carica il Telco e restituisce X_train, X_test, y_train, y_test in NumPy.

    Lo split viene prima del preprocessing: encoder e scaler imparano categorie,
    media e deviazione dal solo train, altrimenti il test entra nel modello.
    """
    df = pd.read_csv(PERCORSO_DATI)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    y = (df["Churn"] == "Yes").astype(int)              # 1 = ha abbandonato
    X = df.drop(columns=["customerID", "Churn", "TotalCharges"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=QUOTA_TEST, random_state=SEME, stratify=y
    )

    numeriche = X.select_dtypes(include="number").columns
    testo = X.columns.difference(numeriche)
    preproc = ColumnTransformer([
        ("testo", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), testo),
        ("numeri", StandardScaler(), numeriche),
    ])
    X_train = preproc.fit_transform(X_train)            # impara SOLO dal train
    X_test = preproc.transform(X_test)

    # y come colonna (n, 1): serve cosi' allineata all'output 1 della rete.
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 2. Struttura: pesi casuali piccoli, bias a zero
# --------------------------------------------------------------------------- #
def inizializza_rete(dim_input: int, dim_nascoste=DIM_NASCOSTE, seme=SEME):
    """Crea i parametri della rete: una lista di layer, ognuno con W e b.

    - Pesi: casuali PICCOLI (randn * 0.01). Piccoli e casuali servono per due motivi.
      Casuali: se tutti i neuroni di un layer partono uguali imparano la stessa cosa
      per sempre (simmetria mai rotta). Piccoli: pesi grandi mandano subito le
      attivazioni nelle zone piatte delle funzioni -> gradienti nulli, non si impara.
    - Bias: a zero. Il bias non soffre di simmetria (lo rompono gia' i pesi),
      quindi zero va benissimo e non introduce sbilanciamenti iniziali.
    """
    rng = np.random.default_rng(seme)
    dimensioni = [dim_input] + list(dim_nascoste) + [1]   # es. [44, 8, 8, 1]

    rete = []
    for dim_entra, dim_esce in zip(dimensioni[:-1], dimensioni[1:]):
        W = rng.standard_normal((dim_entra, dim_esce)) * 0.01
        b = np.zeros((1, dim_esce))
        rete.append({"W": W, "b": b})
    return rete, dimensioni







def main():
    X_train, X_test, y_train, y_test = prepara_dati()
    dim_input = X_train.shape[1]
    rete, dimensioni = inizializza_rete(dim_input)

    print("=== Stadio 0: dati e struttura ===")
    print(f"Train: X {X_train.shape}, y {y_train.shape}")
    print(f"Test:  X {X_test.shape},  y {y_test.shape}")
    print(f"Architettura: {' -> '.join(map(str, dimensioni))}")
    print(f"Attivazioni:  nascosti = ReLU, output = sigmoide (binaria)\n")
    for i, layer in enumerate(rete, start=1):
        print(f"  layer {i}: W {layer['W'].shape}, b {layer['b'].shape}, "
              f"|W| medio = {np.abs(layer['W']).mean():.4f}, "
              f"b tutti zero = {np.all(layer['b'] == 0)}")


if __name__ == "__main__":
    main()
