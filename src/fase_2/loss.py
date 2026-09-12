import numpy as np

def calcola_loss(y_pred, y):
    """Binary cross-entropy media tra predizioni e verita'.

    Formula per un singolo campione (dai concetti):
        loss = -[ y·log(y_pred) + (1-y)·log(1-y_pred) ]
    """
    array_perdite = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(np.mean(array_perdite))
