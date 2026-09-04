# Progetto 1 — Tracking esperimenti

**Dataset:** Telco Customer Churn (7043 righe, ~27% churner)
**Split:** Train 70% / Validation 15% / Test 15% — stratificato
**Costi di business:** C_FP = 170€ (sconto sprecato) · C_FN = 400€ (cliente perso)
**Soglia teorica:** C_FP / (C_FP + C_FN) = 0.298

---

| # | Modello | Feature tolte | Soglia | AUC (test) | Accuracy (test) | Precision (1) | Recall (1) | F1 (1) | FP | FN | Costo test | Note |
|---|---------|--------------|--------|-----------|----------------|--------------|-----------|--------|----|----|-----------|------|
| 1 | LogisticRegression (max_iter=2000) | TotalCharges | 0.400 | 0.820 | 0.772 | 0.56 | 0.64 | 0.60 | 141 | 100 | 63.970€ | Soglia empirica diversa dalla teorica (0.40 vs 0.30) |
| 2 | XGBoost (max_depth=4, lr=0.05, early_stopping=50) | TotalCharges | 0.260 | 0.828 | 0.727 | 0.49 | 0.80 | 0.61 | 233 | 55 | 61.610€ | 98/1000 alberi usati. Soglia empirica vicina alla teorica (0.26 vs 0.30) |
| 3 | | | | | | | | | | | | |

---

## Note per esperimento

### #1 — Logistic Regression baseline
- TotalCharges rimossa: correlazione 0.9996 con tenure × MonthlyCharges
- Soglia empirica (0.40) diverge dalla teorica (0.298) — il modello non è perfettamente calibrato
- Dummy accuracy: 0.735 → il modello aggiunge ~4 punti
- Pesi più influenti: tenure (-0.82), Contract_Two year (-0.72), InternetService_DSL (-0.60)
- AUC train: non misurata (modello semplice, overfitting improbabile)

### #2 — XGBoost (max_depth=4, lr=0.05, early_stopping=50)
- Senza early stopping e max_depth: AUC train 0.993 vs test 0.798 — overfitting severo
- Con max_depth=4 senza early stopping: AUC train 0.953 vs test 0.805 — ancora troppo
- Versione finale: AUC train 0.881 vs test 0.828 — gap 0.053, accettabile
- Early stopping fermato a 98 alberi su 1000: i restanti avrebbero memorizzato rumore
- Vince su AUC (+0.008), costo (-2.360€) e recall (+0.16). Perde su accuracy (-0.045) perché segnala più clienti (233 FP vs 141) — ma il conto economico torna: 45 clienti salvati × 400€ > 92 sconti extra × 170€
- Soglia empirica (0.26) molto più vicina alla teorica (0.298) della LR (0.40) → meglio calibrato
- Feature più influenti: Contract_Month-to-month (0.53), InternetService_Fiber optic (0.14)