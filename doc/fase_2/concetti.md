# Concetti Backpropagation

## Neurone
Un nodo della rete che fa una cosa: prende numeri in input e sputa un output dopo averli combianti in un certo modo.

La combinazione è: 
- **z = (w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ) + b**

Questo valore viene passato alla funzione di attivazione, che può variare, per esempio: 
- **a = relu(z)**


Rendiamolo concreto con un cliente Telco. Immagina un neurone con solo 3 ingressi:
- x = [ contratto_mensile=1,  tenure=-1.2,  MonthlyCharges=0.8 ]
- w = [ 2.0,                  -1.5,          0.3 ]
- b = -0.5
- z = (2.0·1) + (-1.5·-1.2) + (0.3·0.8) + (-0.5)
    =   2.0   +    1.8       +    0.24    -   0.5   =  3.54

Il **peso w** è "quanto conta questo ingresso" — e il segno dice in che direzione:
- w₁ = 2.0 → "avere il contratto mensile spinge FORTE verso churn"
- w₂ = -1.5 → "più mesi di anzianità (tenure alto) spinge CONTRO il churn" (segno negativo)
- w₃ = 0.3 → "la spesa mensile conta poco"

Nota, il segno influisce in questo modo solo se la rete ha un layer solo, ma tendenzialmente non lo è e quel segno in realtà potrebbe spingere in senso opposto (dipende dai segni del layer precedente)

Il **bias b** è la predisposizione del neurone ad accendersi dagli ingressi (z > 0), più è bassa e maggiore dovrà essere la somma dei pesi per compensare e accendere il neurone.

## Forward pass
  
Nell'esercizio 44 → 8 → 8 → 1:
- Layer 1: 8 neuroni, ognuno guarda tutte le 44 feature. Ecco perché W è (44, 8): 44 pesi per neurone × 8 neuroni. Ogni neurone impara un "pezzetto di
pattern" diverso (uno magari si specializza su "cliente nuovo con contratto mensile", un altro su altro).
- Layer 2: 8 neuroni che guardano gli 8 output del layer 1 → W è (8, 8). Combina i pezzetti in pattern più astratti.
- Layer 3 (output): 1 neurone che guarda gli 8 del layer 2 → W è (8, 1), con sigmoide → probabilità di churn.

Il forward pass è semplicemente il processo di propagazione in avanti dei dati, da layer N a layer N+1.

## Loss

La Loss è la misura dell'errore, un unico valore per tutta la rete. Alla rete, e quindi ai singoli nodi, serve avere una misura di quanto sbaglia.

Si confrontano la predizione (y_pred) con la verità (y). Per esempio nella classificazione binaria usaimo la binary cross-entropy:
- loss = -[ y·log(y_pred) + (1-y)·log(1-y_pred) ]

Ovvero:
- Cliente ha davvero churn (y=1), rete dice 0.95 → loss ≈ 0.05 (quasi giusto, punizione minima)
- Cliente ha davvero churn (y=1), rete dice 0.10 → loss ≈ 2.3 (sbaglia con sicurezza, punizione alta)

## Gradient e Backpropagation

La backpropagation è quindi il processo di calcolo del gradiente. Diverso dalla fase di aggiustamento dei pesi dei neuroni, che è successiva.

La domanda è "Per ognuno dei pesi e dei bias, se lo aumento un pochino, la loss sale o scende? E di quanto?" - la risposta è il **gradiente (∂loss/∂w)**

Puoi calcolarla in modo grezzo, letteralmente provando (si chiama differenza finita):
- w = 2.0    → loss = 0.80
- w = 2.01   → loss = 0.83     (ho aumentato w di 0.01, la loss è salita di 0.03)
- pendenza ≈ (0.83 - 0.80) / 0.01 = +3.0

Quel +3.0 è la derivata. Cosa mi dice, letto pezzo per pezzo:
- Il segno +: aumentando w, la loss sale. Quindi per farla scendere devo andare nella direzione opposta →
  diminuire w.
- Il numero 3.0: la sensibilità. La loss cambia 3 volte più in fretta di w. Un valore grande = questo peso ha
  un forte effetto sull'errore, va corretto con decisione.

Nella rete vera non hai un peso, ne hai migliaia (le matrici W e i b). **Il gradiente è semplicemente la lista 
di tutte queste derivate, una per ciascun peso e ciascun bias.**
- gradiente = [ ∂loss/∂w₁,  ∂loss/∂w₂,  ∂loss/∂w₃,  ...,  ∂loss/∂b₁, ... ]

Il simbolo **∂loss/∂w₁** si legge "quanto cambia la loss se muovo w₁ (tenendo fermi tutti gli altri)". È la stessa
domanda di prima, fatta a un peso alla volta.


- ──indietro──►  quanto ha sbagliato ogni neurone del layer 2
- ──indietro──►  quanto ha sbagliato ogni neurone del layer 1
- ──indietro──►  quanto è "colpevole" ogni singolo peso

Ogni passo indietro moltiplica il "senso di colpa" del layer successivo per due cose locali: il peso
attraverso cui è passato il segnale, e la derivata dell'attivazione.

Ottenuti i gradienti, l'aggiornamento è una riga per ogni parametro:
- W = W - learning_rate * gradiente_W
- b = b - learning_rate * gradiente_b
 
Spiegazione:
- Il gradiente punta nella direzione in cui la loss cresce, quindi vai nella direzione opposta per farla scendere (il segno meno).
- Il learning_rate (es. 0.01) è la lunghezza del passo

## Il loop completo

Tutto l'allenamento è questo ciclo, ripetuto per centinaia di epoche:

ripeti tante volte:
1. FORWARD:   dai i dati alla rete, ottieni y_pred      (Stadio 1)
2. LOSS:      misura quanto sbaglia                      (Stadio 2)
3. BACKWARD:  propaga l'errore indietro, calcola i gradienti  (Stadio 3)
4. UPDATE:    W = W - lr·grad,  b = b - lr·grad          (Stadio 3)

A ogni giro i pesi si spostano di un pelo verso valori migliori, la loss scende, y_pred si avvicina alla
  verità. 

Alla fine i pesi non sono più casuali: codificano i pattern del churn imparati dai dati. Quello è il
  modello, ed è quello che avrà senso salvare.

## Vanishing gradient

Il segnale d'errore, tornando indietro strato per strato, si rimpicciolisce a ogni passaggio finché arriva ai primi layer troppo debole per muoverli. 
È il motivo storico per cui per anni non si riuscivano ad allenare reti profonde.

Nell'esercizio la causa era l'inizializzazione randn * 0.01 dei weight.
Guardando le Z della cache: si rimpicciolivano di un fattore ~10 a ogni strato (0.0x → 0.000x → 0.00000x). 
Pesi piccoli → attivazioni piccole → gradienti piccoli → primi layer immobili.

Il rimedio: He initialization. Invece di moltiplicare per un 0.01 fisso a caso, scali i pesi in base a quanti ingressi ha quel layer, così la varianza del segnale resta costante attraverso gli strati invece di collassare.
- W = randn(dim_entra, dim_esce) * sqrt(2 / dim_entra)