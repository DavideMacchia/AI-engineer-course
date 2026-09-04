"""Caricamento, pulizia e divisione dei dati.

Lo split viene PRIMA del preprocessing (vedi preprocessing.py): encoder e scaler
devono imparare solo dal train, altrimenti la struttura del test entra nel modello.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PERCORSO_DATI, SEME, QUOTA_TEMP, QUOTA_TEST


def carica_e_pulisci() -> pd.DataFrame:
    """Carica il CSV e sistema TotalCharges (testo perché 11 righe hanno la cella vuota)."""
    df = pd.read_csv(PERCORSO_DATI)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna()


def separa_X_y(df: pd.DataFrame):
    """Separa la risposta (y) dalle domande (X).

    TotalCharges esce: è quasi esattamente tenure * MonthlyCharges (correlazione 0.9996),
    quindi non aggiunge informazione e confonde i pesi delle due colonne da cui deriva.
    """
    y = (df["Churn"] == "Yes").astype(int)   # 1 = ha abbandonato, 0 = è rimasto
    X = df.drop(columns=["customerID", "Churn", "TotalCharges"])
    return X, y


def dividi(X, y):
    """Divide in train / validation / test, stratificato.

    Train      — ci alleni il modello
    Validation — ci scegli la soglia (e ogni altra decisione: iperparametri, feature, modello)
    Test       — ci misuri il risultato finale, una volta sola, quando hai già deciso tutto
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=QUOTA_TEMP, random_state=SEME, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=QUOTA_TEST, random_state=SEME, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
