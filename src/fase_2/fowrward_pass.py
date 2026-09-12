from typing import Any
from attivazione import relu, sigmoid

def forward_pass(X: Any, rete: list[Any]):
    A = X                       # attivazione di partenza: i dati stessi (A_0)
    ultimo_layer = len(rete) - 1
    for i, layer in enumerate(rete):
        W = layer["W"]
        b = layer["b"]
        Z = A @ W + b           # combinazione lineare del layer
        if i == ultimo_layer:
            A = sigmoid(Z)      # output: probabilita' di churn in (0, 1)
        else:
            A = relu(Z)         # layer nascosti
    return A
