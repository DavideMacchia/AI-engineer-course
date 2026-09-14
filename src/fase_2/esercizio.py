"""Fase 2 — Stadio 0: dati e struttura di una rete neurale da zero (NumPy).

La traccia originale usa make_moons (2 feature -> architettura 2 -> 8 -> 8 -> 1).
Qui il dataset e' il Telco: l'input NON e' 2, e' il numero di colonne dopo il
preprocessing (one-hot sulle categoriche + scaling sulle numeriche) = 44.
Quindi l'architettura diventa  44 -> 8 -> 8 -> 1: cambia solo la dimensione
d'ingresso, i due hidden layer da 8 e l'output da 1 restano identici.

In questo stadio si prepara SOLO l'impalcatura: dati divisi e pesi inizializzati.
Il forward pass, la loss e la backpropagation arrivano negli stadi successivi.
"""

from back_propagation import backward
from data import prepara_dati, inizializza_rete
from evaluation import soglia_soglia_ottima, costo_sul_test
from loss import calcola_loss
from update import aggiorna
from forward_pass import forward_pass

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = prepara_dati()
    print(f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"churner nel test: {int(y_test.sum())}")   # <-- aggiungi questa

    dim_input = X_train.shape[1]
    rete, dimensioni = inizializza_rete(dim_input)

    LEARNING_RATE = 0.01
    EPOCHE = 1000
    for epoca in range(EPOCHE):
        y_pred, cache = forward_pass(X_train, rete)

        if epoca % 100 == 0:
            loss = calcola_loss(y_pred, y_train)
            print(f"Epoca: {epoca}, Loss: {loss:.4f}")

        gradienti = backward(y_pred, y_train, cache, rete)
        rete = aggiorna(rete, gradienti, LEARNING_RATE)

    soglia, griglia, costi = soglia_soglia_ottima(X_val, y_val, rete)
    print(f"Soglia OTTIMA: {soglia:.4f}")

    costo_finale = costo_sul_test(X_test, y_test, soglia, rete)
    print(f"Costo sul test: {costo_finale} €")

if __name__ == "__main__":
    main()
