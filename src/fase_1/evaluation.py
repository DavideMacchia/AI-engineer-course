"""Report dei risultati: scelta soglia, metriche finali, sguardo dentro il modello."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
)

from threshold import soglia_teorica, costo
from models import importanze


def report_soglia(y_val, prob_val, soglia_scelta, griglia, costi, costo_fp, costo_fn):
    idx_05 = int(np.argmin(np.abs(griglia - 0.5)))
    print("=== Scelta della soglia (validation) ===")
    print(f"Soglia teorica  C_FP / (C_FP + C_FN) = {soglia_teorica(costo_fp, costo_fn):.3f}")
    print(f"Soglia empirica (minimo costo)       = {soglia_scelta:.3f}")
    print(f"Costo sul validation a soglia scelta = {min(costi)}€")
    print(f"Costo sul validation a soglia 0.5    = {costi[idx_05]}€")


def report_test(y_test, probabilita, soglia_scelta, costo_fp, costo_fn):
    """Test: busta chiusa, aperta una volta sola, con la soglia già decisa."""
    predizioni = (probabilita >= soglia_scelta).astype(int)

    print("\n=== Risultato finale (test) ===")
    print(f"ROC-AUC:     {roc_auc_score(y_test, probabilita):.3f}")
    print(f"Accuratezza: {accuracy_score(y_test, predizioni):.3f}")
    # Il numero significa qualcosa? Confronto con chi dice sempre "nessuno abbandona"
    print(f"Se dicessi sempre 'resta' avrei: {(y_test == 0).mean():.3f} di accuratezza")

    tn, fp, fn, tp = confusion_matrix(y_test, predizioni).ravel()
    print(f"Costo sul test: {costo(y_test, probabilita, soglia_scelta, costo_fp, costo_fn)}€ "
          f"({fp} FP, {fn} FN)")

    print(confusion_matrix(y_test, predizioni))
    print(classification_report(y_test, predizioni))


def report_importanze(model, nomi_colonne):
    """I pesi/importanze che il modello ha imparato, se li espone."""
    pesi = importanze(model, nomi_colonne)
    if pesi is None:
        print("\n(Questo modello non espone pesi né importanze delle feature.)")
        return
    if hasattr(model, "intercept_"):
        print(f"Intercetta: {model.intercept_[0]:+.2f}")
    print("\nLe 10 feature più influenti:")
    print(pesi.reindex(pesi.abs().sort_values(ascending=False).index).head(10).round(4).to_string())
