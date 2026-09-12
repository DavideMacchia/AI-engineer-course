import numpy as np

# --------------------------------------------------------------------------- #
# Attivazioni (definite qui, si useranno nel forward pass dello stadio dopo)
# --------------------------------------------------------------------------- #
def relu(z):
    """Attivazione dei layer nascosti: semplice, niente zone piatte per z > 0,
    quindi i gradienti passano bene. (Alternativa classica: tanh.)"""
    return np.maximum(0.0, z)


def sigmoid(z):
    """Attivazione dell'OUTPUT. Questa scelta NON e' libera: e' classificazione
    binaria, serve una probabilita' in [0, 1] per la classe 1 (churn).
    La sigmoide schiaccia qualsiasi numero reale in (0, 1) -> e' esattamente
    quella. (Con piu' di 2 classi userei softmax; per una regressione, nessuna.)
    """
    return 1.0 / (1.0 + np.exp(-z))