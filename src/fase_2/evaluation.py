import sys
from pathlib import Path

# fase_1 e fase_2 si lanciano dall'interno della propria cartella (import piatti).
# Per riusare scegli_soglia di fase_1 aggiungo src/ al path: cosi' fase_1 e'
# raggiungibile come namespace package (from fase_1.threshold import ...).
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fase_1.threshold import scegli_soglia, costo
from forward_pass import forward_pass

COSTO_FP = 170   # sconto offerto a chi sarebbe rimasto comunque
COSTO_FN = 400   # cliente perso senza aver fatto niente

# costo = fp * COSTO_FP + fn * COSTO_FN
SOGLIA_TEORICA = COSTO_FP / (COSTO_FP + COSTO_FN)

def soglia_soglia_ottima(X_val, y_val, rete):
    # scegli la soglia che minimizza il costo sul val
    prob_val, _ = forward_pass(X_val, rete)
    return scegli_soglia(y_val, prob_val, COSTO_FP, COSTO_FN)

def costo_sul_test(X_test, y_test, soglia_ottima, rete):
    prob_test, _ = forward_pass(X_test, rete)
    return costo(y_test, prob_test, soglia_ottima, COSTO_FP, COSTO_FN)
