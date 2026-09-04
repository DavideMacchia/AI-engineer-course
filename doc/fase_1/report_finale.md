# Il problema

Nel contesto Telco Customer Churn, un operatore perde soldi quando un cliente lascia e cambia contratto. Esiste la possibilità che questi non lascino se intercettati e gli venga proposto un rinnovo a prezzo migliore. Il rischio è che si mandi l'offerta anche a chi non ha intenzione di cambiare.
Il progetto è un classificatore ML che serve per predire quali clienti sono in procinto di cambiare fornitore, intercettarli e offrire loro un rinnovo per tenerli dentro.
Il costo di un cliente perso è stato stimato a 400 euro, il costo di un cliente a cui è stato offerto un rinnovo ma non voleva lasciare è stimato a 170 euro. 

# Il Risultato

Il modello tenta di minimizzare i costi predendo quali clienti stanno lasciando.
Ciò che conta maggiormente sono i Falsi Positivi (FP), i Falsi Negativi (FN) e i corrispondenti costi (170 e 400 euro).

Il costo se non si mandassero offerte a nessuno è di 280 FN × 400€ = 112.000€
Il costo se mandassimo offerte a tutti è di 775 FP × 170€ = 131.750€

Il costo della predizione effettuata dal Logistic Classifier è di 63970 euro.
Il costo della predizione effettuata da XGBoost è di 61610 euro.

XGBoost è marginalmente migliore (-2.360€, 3,7%) rispetto al Logical Classifier. Il guadagno vero è rispetto al non fare nulla: -50.390€.
XGBoost ha accuracy peggiore (0.727 vs 0.772) ed è comunque il modello migliore.

# Le Decisioni

- TotalCharges rimossa perché TotalCharges ≈ tenure × MonthlyCharges, correlazione 0.9996. Quindi non avrebbe aggiunto informazioni e anzi, ha aggiunto accuracy.
- La soglia di default è 0.5, non è una soglia ottima e quindi è stata calcolata ipotizzando i costi di FP e FN con la seguente formula: **costo_fp / (costo_fp + costo_fn)**
- Il triple split è fondamentale per avere una parte di dataset dedicata alla validazione, questa altrimenti dovrebbe avvenire sulla porzione di dataset del training e quindi scegliere la soglia (validazione) sarebbe come barare ed avere una stima ottimistica.
- L'XGBoost vince sul Logical classifier perché ha ridotto i FN inviando più offerte di rinnovo, facendo calare l'accuracy perché sono aumentati gli FP ma compensando a livello di costi e quindi rendendolo conveniente.
- L'overfitting è stato domato introducendo max_depth=4, learning_rate=0.05 ed early_stopping_rounds=50 nella fase di training.


# Cosa manca / limiti
I costi C_FP e C_FN sono inventati, non validati con un business reale, quindi il modello può dare esclusivamente un'indicazione a livello generico di come potrebbe funzionare in quel contesto.

Cross-validation assente. Soglia e valutazione dipendono da un singolo split con SEME=42. La differenza di 2.360€ tra i due modelli potrebbe non reggere con un seme diverso. È esattamente l'obiezione che ti farebbero in colloquio.

Modello non serializzato. Non c'è joblib.dump. Il preprocessore, il modello e la soglia esistono solo durante l'esecuzione. Nessun servizio può usarli.

Nessuna validazione temporale. Lo split è casuale, ma il churn è un fenomeno che evolve nel tempo. In produzione allenaresti sui dati vecchi e prediresti sui nuovi — uno split casuale può essere ottimista rispetto a quello scenario.

# Come riprodurre

**Requisiti:** Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone <url-del-repo>
cd ai_course
uv sync
```

**Dataset:** scarica `WA_Fn-UseC_-Telco-Customer-Churn.csv` da [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e mettilo in `data/`.

**Esecuzione:**

```bash
cd src/fase_1
python esercizio.py logistic    # baseline
python esercizio.py xgboost     # modello scelto
python esercizio.py dummy       # baseline stupida, per confronto
```

**Esperimenti tracciati:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Poi apri `http://localhost:5000`.

**Parametri:** costi di business, seme e proporzioni dello split stanno in `src/fase_1/config.py`.