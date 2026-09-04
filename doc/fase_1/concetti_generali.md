# Fase 1 — Concetti consolidati

## Risorse
- [Calude chat](https://claude.ai/chat/618e04e5-e2e1-4132-9e2b-ec9d28a12d8e)
- [Kaggle Course - Intro](https://www.kaggle.com/code/dansbecker/how-models-work/tutorial)
- [Kaggle Course - Intermediate](https://www.kaggle.com/learn/intermediate-machine-learning)

## 1. Accuracy su dataset sbilanciati

L'accuracy misura quante volte il modello azzecca la classificazione sul totale: (TP + TN) / totale. Su un dataset dove il 73% dei clienti resta (classe 0), un modello che risponde sempre "resta" senza guardare niente ottiene 0.734. Il nostro modello con 30 feature fa 0.793 — solo 6 punti in più. L'accuracy non stava dicendo "il modello è bravo al 79%", stava dicendo "di cui 73 li regala la composizione del dataset". Il valore reale del modello sta in quei 6 punti.

## 2. Precision, Recall e F1-score

- **Precision** = TP / (TP + FP) = 197 / 311 = 0.63. 
Di tutti quelli che il modello ha segnalato come churner, quanti lo erano davvero. Precision bassa = molti falsi positivi (sconti sprecati).
- **Recall** = TP / (TP + FN) = 197 / 374 = 0.53. Di tutti quelli che erano davvero churner, quanti ne ha presi. Recall bassa = molti falsi negativi (clienti persi senza intervento).


La differenza sta nel denominatore: precision parte dalla predizione (colonna), recall parte dalla realtà (riga).

Entrambe ignorano i veri negativi (i 919 facili) e costringono a guardare solo la classe che interessa.

- **F1-score** = 2 × (precision × recall) / (precision + recall). è una sintesi di precision e recall in un numero solo. F1 tende allo 0 maggiore è la distanza tra i due. Serve guardarlo quando non puoi calcolare la soglia ottimale dai costi perché nessuno ti sa dire quanto costa un FP e quanto un FN. F1 è il compromesso di default.

## 3. AUC e indipendenza dalla soglia

AUC misura la qualità dell'ordinamento prodotto dal modello. Operativamente: prendi un churner e un non-churner a caso; AUC = la probabilità che il modello assegni un punteggio più alto al churner. Il nostro 0.830 significa 83 coppie su 100 ordinate correttamente. 0.5 = ordine casuale (il dummy), 1.0 = ordine perfetto.

Non dipende dalla soglia perché i punteggi di `predict_proba` sono calcolati una volta sola dal modello e non cambiano mai. La soglia è un taglio applicato dopo, sui punteggi già esistenti. Spostarla cambia precision e recall ma non cambia l'ordinamento — quindi non cambia l'AUC.

## 4. Soglia ottimale derivata dai costi

Il default di `predict()` taglia a 0.5, ma quella soglia è corretta solo quando i due errori costano uguale.

La soglia ottimale si ricava dal punto di pareggio tra il costo atteso di non agire e il costo atteso di agire:

```
p * C_FN = (1 - p) * C_FP
```

Da cui:

```
p* = C_FP / (C_FP + C_FN)
```

Con C_FP = 170€ (sconto sprecato) e C_FN = 400€ (cliente perso): p* = 170 / 570 = 0.298. Confermato dalla griglia empirica, che aveva il minimo a 0.30.

La formula non contiene il modello: la soglia ottimale dipende solo dai costi di business, non dalla bravura del classificatore.

## 5. Perché servono tre set e non due

- **Train** — ci alleni il modello.
- **Validation** — ci scegli la soglia (e qualsiasi altra decisione: iperparametri, feature, tipo di modello).
- **Test** — ci misuri il risultato finale, una volta sola, a busta chiusa.

Se usi il test set per prendere decisioni (come scegliere la soglia), la stima che riporti è ottimista: hai ottimizzato sugli stessi dati che dovevano giudicarti.

## 6. `get_dummies` prima dello split

Fare `get_dummies` su tutto il dataset prima dello split fa entrare la struttura del test set nel train: il modello sa quali valori categorici esistono nel test. In produzione quella garanzia non esiste — un valore nuovo crea una colonna in più, un valore assente ne toglie una, e `predict` esplode.

La soluzione è `OneHotEncoder` fittato solo sul train, con `handle_unknown="ignore"`: un valore mai visto diventa una riga di zeri invece di un errore.

## 7. Overfitting, Underfitting, Random forest

Rispettivamente sono due fenomeni che diminuiscono l'accuratezza del modello, entrambe causate dalla fase di training. 

Nel primo caso (**Overfitting**) il modello è stato addestrato a catalogare troppo precisamente, imparando dettagli inutili (rumore del training set) invece del segnale. 
Identificare anche pattern ampi, ma ci aggiunge pattern falsi che esistono solo nel train. 
Il test te lo dice: se accuracy sul train è 0.95 e sul test è 0.79, il modello sta overfittando.

**Underfitting** quando il modello fallisce nel classificare adeguatamente perché è tropo semplice e quindi non è riuscito a catturare la relazione tra le feature e il target (le features possono anche essere centinaia ma comunque non bastare). 
L'underfitting è quando il modello non riesce a catturare il segnale nemmeno sul training set — performa male su tutto, train e test. 
La causa tipica è un modello troppo semplice per il problema: una retta dove serviva una curva.

Nei **Decision Trees** si ovvia ai due problemi descritti usando la tecnica chiamata **Random Forest** che consiste nel prendere la decisione usando tanti Decision Tree la randomizzazione è su righe (bootstrap) e feature (a ogni nodo, non a ogni albero). I singoli alberi overfittano quasi tutti, ma ognuno memorizza rumore diverso.
Il risultato finale deciso è un'aggregazione di tutti i risultati ottenuti dai vari alberi che cancella il rumore, per esempio il risultato che compare il 50% delle volte oppure una media di risultati numerici.

## 8. Variabili categoriche

Variabili con un numero limitato di valori distinti (es. Contract: Month-to-month, One year, Two year). Tre strategie di encoding:

Drop: se la colonna non è utile alla predizione, si elimina.
Ordinal encoding: si assegna un numero a ogni valore (1, 2, 3). Funziona quando l'ordine esiste nel dato (durata del contratto, taglia S/M/L). Sbagliato quando l'ordine non esiste (InternetService: DSL, Fiber optic, No) — il modello lineare tratta quei numeri come una scala e il coefficiente non ha senso. I decision tree sono meno sensibili al problema, perché provano tutti i tagli indipendentemente dall'ordine numerico.
One-hot encoding: ogni valore diventa una colonna a sé (0 o 1). Non assume ordinamento, ma il numero di colonne esplode se i valori distinti sono molti (codice postale, ID prodotto). In quel caso si usa il target encoding — ogni valore viene sostituito con la media del target per quel valore (es. il churn rate medio di quel codice postale). Calcolato sul train, mai su tutto il dataset, altrimenti è leakage.

## 9. Pipeline

Una pipeline raggruppa preprocessing e modello in un oggetto unico. Il vantaggio principale non è la pulizia del codice — è che fit e transform vengono applicati automaticamente nell'ordine giusto, rendendo impossibile il data leakage accidentale (encoder o scaler fittati sul test). Secondo vantaggio: in produzione salvi e carichi un oggetto solo invece di tenere sincronizzati preprocessore, modello e soglia separatamente.

## 10. Cross-validation

Processo di divisione del dataset (tolta una parte fissa per il test finale) in più parti che vengono usate una alla volta per la validazione mentre il restante dataset per il training. 

Il processo permette di avere una misura della qualità del modello che si basa su tutte le parti del dataset e non solo una. La tecnica aumenta il tempo di esecuzione e và usata nei casi in cui il dataset con validation set fisso sarebbe troppo ridotto per dare risultati affidabili.

## 11. Gradient Boosting

Processo iterativo che costruisce un ensemble aggiungendo un modello a ogni iterazione:

Il primo modello fa predizioni, probabilmente inaccurate (es. predice 0.27 per tutti).
Si calcola il residuo per ogni cliente: valore vero meno predizione. Può essere positivo (il modello sottovaluta) o negativo (il modello sovrastima).
Si aggiunge un nuovo albero che impara a predire i residui, non il churn. Vede le stesse feature di sempre (tenure, Contract, ecc.) — quello che cambia è il target: residui invece del valore originale. È così che scopre per chi correggere.
La predizione dell'ensemble è la somma di tutti i modelli: predizione di base + correzione 1 + correzione 2 + ... Ogni modello aggiunto è una patch su quello che restava da correggere.
Si reitera: i residui si ricalcolano sulla somma attuale, il modello successivo corregge quelli.

Differenza col Random Forest: nel RF gli alberi sono indipendenti e votano (riduce il rumore per media). Nel Gradient Boosting sono sequenziali e si correggono a vicenda (riduce l'errore per correzione iterativa).

## 12. Data Leakage

Fenomeno in cui il training data contiene informazioni che non saranno disponibili in produzione. Questo porta il modello a performare bene in fase di testing ma non in produzione.
Ci sono due tipi: Target leakage e train-set contamination.

**Target leakage**: quando un dato esiste nel dataset iniziale perchè conseguenza di un altro dato ma in produzione non esiste ancora: esempio valore catastale immobile, valore di vendita effettivo e il modello che cerca di predire in quanto tempo verrà venduto. In produzione il valore di vendita manca per forza di cose.

**Train-test contamination**: qualsiasi informazione dal validation o test set che influenza il training — nella struttura, nei parametri del preprocessing, o nella selezione delle feature

# 13. Regolarizzazione

L'impostazione di limiti durante il training, anche a costo di peggiorarne i risultati sul training set, affinchè ci sia un guadagno di prestazioni sui dati muovi. Esempio max_depth=4 su XGBoost che prima dell'introduzione overfittava il training set ma aveva scarsi risultati sui nuovi dati. 
Altri parametri usati con XGBoost: 

- max_depth=4 — limita la profondità di ogni albero
- learning_rate=0.05 — ogni correzione pesa il 5% invece del 100%, il modello impara più lentamente
- early_stopping_rounds=50 — ferma l'addestramento quando il validation smette di migliorare (98 alberi su 1000 massimi)