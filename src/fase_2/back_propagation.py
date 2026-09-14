
def backward(y_pred, y, cache, rete):
    """Restituisce i gradienti: per ogni layer, un dW e un db."""
    # dZ = dA * (Z > 0)          # solo per i nascosti; per l'output dZ = y_pred - y
    # dW = A_prev.T @ dZ / n
    # db = dZ.sum(axis=0, keepdims=True) / n
    # dA_prev = dZ @ W.T         # l'errore da passare al layer precedente → diventa il dA del prossimo giro

    n = y.shape[0]
    gradienti = [None] * len(rete)
    dA = None

    # loop parte dal fondo a ritroso per ogni layer
    for i in reversed(range(len(rete))):
        Z = cache[i]["Z"]
        A_prev = cache[i]["A_prev"]
        W = rete[i]["W"]

        if i == len(rete) - 1:
            dZ = y_pred - y  # ultimo layer (primo iterato): il regalo, niente dA
        else:
            dZ = dA * (Z > 0)  # nascosti: dA arriva da valle, derivata ReLU

        # dato dZ (l'errore sulla combinazione lineare di questo layer)
        # dW = A_prev.T @ dZ / n si legge: quanto ha inciso W (i pesi di questo layer) nell'errore, dato dZ.
        # E A_prev compare nel calcolo perché la colpa di un peso dipende da quanto era grande l'ingresso che lo ha attraversato.
        # "/n" è la loss è una media su n esempi. Riallinea il gradiente al fatto che la loss era mediata
        dW = A_prev.T @ dZ / n

        # Nel forward Z = A_prev @ W + b: il bias si somma e basta, non moltiplica niente. La derivata di una somma rispetto a uno dei suoi termini è 1.
        # Quindi la colpa del bias è semplicemente la colpa dell'uscita, dZ.
        # Ma dZ è (n, 8) — un valore per ogni esempio, per ogni neurone. Il bias invece è (1, 8) — uno per neurone, condiviso da tutti gli esempi. Come passi da n valori a uno? Li sommi sugli esempi. È quello che fa axis=0: somma lungo le righe (gli esempi), lasciando gli 8 neuroni. Risultato (1, 8), la forma di b. Il keepdims=True serve solo a tenere la forma (1, 8) invece che collassarla a (8,) — dettaglio tecnico perché il broadcasting dopo funzioni. E /n per lo stesso motivo di prima: media, non somma.
        db = dZ.sum(axis=0, keepdims=True) / n

        # Il ragionamento: nel forward, A_prev (l'output del layer precedente) è entrato in questo layer attraverso W per produrre Z.
        # Quindi la colpa di A_prev è la colpa di Z (cioè dZ) fatta rimbalzare indietro attraverso gli stessi W. Rimbalzare indietro = moltiplicare per W.T.
        dA = dZ @ W.T
        gradienti[i] = {"dW": dW, "db": db}

    return gradienti