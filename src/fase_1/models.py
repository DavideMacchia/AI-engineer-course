"""Registry dei modelli: aggiungerne uno = una riga qui, niente altro da toccare.

Ogni voce è una factory (funzione senza argomenti) che restituisce un modello nuovo.
Tutti i modelli sono classificatori scikit-compatibili con predict_proba: il resto
della pipeline (soglia, valutazione) non deve sapere quale sia in uso.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from config import SEME

# nome -> factory. Per provare un modello nuovo, aggiungi qui la sua riga.
MODELLI = {
    "logistic": {
        "factory": lambda: LogisticRegression(max_iter=2000),
        "fit_kwargs": {},
    },
    "xgboost": {
        "factory": lambda: XGBClassifier(eval_metric="logloss",
                                     random_state=SEME,
                                     max_depth=4,
                                     n_estimators=1000,
                                     learning_rate=0.05,
                                     early_stopping_rounds=50,
                                     ),
        "fit_kwargs": lambda X_val, y_val: {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    },
    "random_forest": lambda: RandomForestClassifier(random_state=SEME),
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
}


def crea_modello(nome: str):
    if nome not in MODELLI:
        disponibili = ", ".join(MODELLI)
        raise ValueError(f"Modello '{nome}' sconosciuto. Disponibili: {disponibili}")
    return MODELLI[nome]["factory"]()


def probabilita_positiva(model, X) -> np.ndarray:
    """P(churn) in modo uniforme per qualunque classificatore."""
    return model.predict_proba(X)[:, 1]


def importanze(model, nomi_colonne) -> pd.Series | None:
    """Pesi/importanze delle feature, se il modello li espone.

    LogisticRegression -> coef_ (con segno: + spinge verso 'abbandona', - verso 'resta').
    XGBoost / RandomForest -> feature_importances_ (magnitudine, senza segno).
    Modelli che non espongono nulla (es. Dummy) -> None.
    """
    if hasattr(model, "coef_"):
        return pd.Series(model.coef_[0], index=nomi_colonne)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=nomi_colonne)
    return None

def allena(model, X_train, y_train, X_val=None, y_val=None):
    """Fit uniforme: passa eval_set solo ai modelli che lo supportano (XGBoost)."""
    if isinstance(model, XGBClassifier) and X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model