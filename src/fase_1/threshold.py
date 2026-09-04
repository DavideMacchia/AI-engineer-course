"""Scelta della soglia dai costi di business, non dal modello."""
import numpy as np
from sklearn.metrics import confusion_matrix


def soglia_teorica(costo_fp: float, costo_fn: float) -> float:
    """p* dal punto di pareggio dei costi: p* = C_FP / (C_FP + C_FN)."""
    return costo_fp / (costo_fp + costo_fn)


def costo(y_vero, prob, soglia, costo_fp, costo_fn) -> int:
    """Costo totale (€) applicando una data soglia."""
    tn, fp, fn, tp = confusion_matrix(y_vero, (prob >= soglia).astype(int)).ravel()
    return fp * costo_fp + fn * costo_fn


def scegli_soglia(y_val, prob_val, costo_fp, costo_fn, griglia=None):
    """Cerca sul VALIDATION la soglia di costo minimo. Restituisce (soglia, griglia, costi).

    La soglia è una decisione, e le decisioni non si prendono sul test.
    """
    if griglia is None:
        griglia = np.arange(0.05, 0.95, 0.01)
    costi = [costo(y_val, prob_val, s, costo_fp, costo_fn) for s in griglia]
    soglia = griglia[int(np.argmin(costi))]
    return soglia, griglia, costi
