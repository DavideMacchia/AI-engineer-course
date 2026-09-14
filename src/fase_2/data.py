import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Split: Train 70% / Validation 15% / Test 15% (stratificato)
QUOTA_TEMP = 0.3   # val + test insieme
QUOTA_TEST = 0.5   # metà di temp -> 15% del totale

CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
PERCORSO_DATI = CARTELLA_PROGETTO / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

SEME = 42
# Architettura: input calcolato dai dati, poi 8 -> 8 -> 1 come da traccia.
DIM_NASCOSTE = [32,16,8]

# --------------------------------------------------------------------------- #
# 1. Dati: carico, pulisco, divido PRIMA di preprocessare
# --------------------------------------------------------------------------- #
def prepara_dati():
    """Carica il Telco e restituisce X_train, X_val, X_test, y_train, y_val, y_test in NumPy.

    Lo split viene prima del preprocessing: encoder e scaler imparano categorie,
    media e deviazione dal solo train, altrimenti il test entra nel modello.
    """
    df = pd.read_csv(PERCORSO_DATI)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    y = (df["Churn"] == "Yes").astype(int)              # 1 = ha abbandonato
    X = df.drop(columns=["customerID", "Churn", "TotalCharges"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=QUOTA_TEMP, random_state=SEME, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=QUOTA_TEST, random_state=SEME, stratify=y_temp
    )
    numeriche = X.select_dtypes(include="number").columns
    testo = X.columns.difference(numeriche)
    preproc = ColumnTransformer([
        ("testo", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), testo),
        ("numeri", StandardScaler(), numeriche),
    ])
    X_train = preproc.fit_transform(X_train)            # impara SOLO dal train
    X_val = preproc.transform(X_val)
    X_test = preproc.transform(X_test)

    # y come colonna (n, 1): serve cosi' allineata all'output 1 della rete.
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    return X_train, X_val, X_test, y_train, y_val, y_test


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
        W = rng.standard_normal((dim_entra, dim_esce)) * np.sqrt(2 / dim_entra)
        b = np.zeros((1, dim_esce))
        rete.append({"W": W, "b": b})
    return rete, dimensioni

