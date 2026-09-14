def aggiorna(rete, gradienti, lr = 0.1):
    # W = W - lr * dW
    # b = b - lr * db
    for layer, g in zip(rete, gradienti):
        layer["W"] = layer["W"] - lr * g["dW"]
        layer["b"] = layer["b"] - lr * g["db"]

    return rete