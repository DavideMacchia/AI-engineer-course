from typing import Any
from attivazione import relu, sigmoid

def forward_pass(X: Any, rete: list[Any]):
    A = X                       # attivazione di partenza: i dati stessi (A_0)
    cache = []                  # un elemento per layer, servira' alla backprop
    ultimo_layer = len(rete) - 1
    for i, layer in enumerate(rete):
        W = layer["W"]
        b = layer["b"]
        A_prev = A              # l'attivazione che ENTRA in questo layer
        Z = A_prev @ W + b      # combinazione lineare del layer
        if i == ultimo_layer:
            A = sigmoid(Z)      # output: probabilita' di churn in (0, 1)
        else:
            A = relu(Z)         # layer nascosti
        cache.append({"A_prev": A_prev, "Z": Z})
    return A, cache
