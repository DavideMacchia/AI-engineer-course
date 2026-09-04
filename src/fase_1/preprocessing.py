"""Costruzione del preprocessore: testo -> 0/1, numeri sulla stessa scala."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def costruisci_preprocessore(X) -> ColumnTransformer:
    """Un ColumnTransformer che tratta separatamente colonne di testo e numeriche.

    Solo le colonne di testo passano dall'encoder: tenure e MonthlyCharges sono già
    numeri e one-hot li spezzerebbe in una colonna per ogni valore distinto.
    handle_unknown="ignore": una categoria mai vista nel train diventa una riga di
    zeri invece di far esplodere predict.
    """
    colonne_numeriche = X.select_dtypes(include="number").columns
    colonne_testo = X.columns.difference(colonne_numeriche)

    return ColumnTransformer([
        ("testo", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), colonne_testo),
        ("numeri", StandardScaler(), colonne_numeriche),
    ])
