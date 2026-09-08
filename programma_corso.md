# Percorso AI Engineer — v2
 
**12 mesi · 5–10 ore a settimana · da software engineer web a AI engineer applicativo mid-level**
 
> Versione riallocata sul profilo di partenza: software engineer mid/senior, 5 anni di web development, laurea in informatica. Obiettivo dichiarato: **mid-level applicativo entro 12 mesi**, con la porta aperta verso il model engineering a mese 18–24, da dentro un lavoro.
 
---
 
## Il profilo target
 
"AI Engineer" oggi sono almeno tre mestieri diversi:
 
1. **Applicativo / prodotto** — costruisce applicazioni LLM affidabili: RAG, agenti, evals, integrazione. Decine di migliaia di posizioni in Europa.
2. **Model / ML engineer** — training, post-training, ottimizzazione dell'inferenza. Poche decine di organizzazioni in Europa, selezione prevalentemente su PhD e pubblicazioni.
3. **Platform / LLMOps** — infrastruttura, serving, GPU, gateway, FinOps dell'inferenza. Nicchia in crescita rapida.
 
**Questo percorso costruisce il profilo 1**, con una deviazione deliberata verso l'open-weight applicato che apre la strada al profilo 2 senza costare tempo.
 
### Perché non anticipare il model engineering
 
Il mercato europeo del training vero è concentrato in poche organizzazioni (Mistral, Kyutai, LightOn, H, Poolside a Parigi; DeepMind e Meta tra Londra e Parigi; Black Forest Labs; Silo AI/AMD a Helsinki; i lab corporate Nvidia/Apple/Microsoft tra Zurigo e Cambridge; il Barcelona Supercomputing Center col progetto ALIA/Salamandra). Il rapporto di domanda contro il profilo applicativo è di ordini di grandezza.
 
La porta d'ingresso realistica al mondo dei modelli, per chi viene dal software engineering, **non è la ricerca ma la training infrastructure / inference engineering**: sistemi distribuiti, performance, pipeline dati su larga scala. Ci si arriva meglio da mid applicativo che da qualsiasi scorciatoia.
 
Nel frattempo, GDPR, AI Act e sovranità del dato stanno spingendo banche, sanità, difesa e settore pubblico europei verso modelli open-weight ospitati in casa. Questo crea una fascia intermedia molto richiesta — fine-tuning applicato, quantizzazione, serving con vLLM, valutazione di modelli open — tecnicamente a metà strada e commercialmente abbondante. **È lì che puntano la Fase 2 e la Fase 4 di questa versione.**
 
---
 
## Budget realistico
 
A 8 ore settimanali medie su 48 settimane sono circa **380 ore**. La versione integrale di questo programma ne richiederebbe ~550. La differenza è stata assorbita con scelte esplicite, non nascoste:
 
- Fase 1 ridotta: l'ML classico serve per la mentalità e per i colloqui, non a livello di padronanza.
- Fase 2 tenuta a livello di **alfabetizzazione** sui transformer più **un fine-tuning fatto bene**. Non è competenza di training e non va spacciata per tale.
- Fase 4 comprimibile in base al tuo background backend.
- **Fase 3 intoccabile e anzi espansa**: è la fase che ti fa assumere.
 
Le durate sono stime. Contano le milestone, non il calendario.
 
---
 
## Le regole del metodo
 
1. **70% pratica, 30% teoria.** Ogni concetto studiato va tradotto in codice entro pochi giorni, altrimenti evapora.
2. **Una sola risorsa principale per fase.** Il "tutorial hell" è il rischio numero uno di chi studia da solo: si sceglie, si finisce, si passa oltre.
3. **Ogni fase si chiude con un progetto pubblico su GitHub** e **un articolo tecnico**. Il portfolio È il corso.
4. **Usa l'AI per studiare più veloce, mai per studiare al posto tuo.** L'AI spiega *dopo* che hai tentato, rivede *dopo* che hai scritto, non scrive mai il codice core. → vedi `metodo-di-studio.md`
5. **Almeno un progetto deve avere un utente vero** che non sei tu. Requisito obbligatorio, non opzionale. → vedi sotto.
 
---
 
## Il requisito dell'utente vero
 
Quattro progetti auto-assegnati sono quattro progetti auto-assegnati. Un solo sistema usato da qualcuno che non sei tu vale più degli altri tre messi insieme, perché porta i vincoli che non ti sai inventare: requisiti che cambiano, dati incoerenti, budget, feedback negativo, gente che usa il prodotto nel modo sbagliato.
 
Candidati validi: un'associazione, uno studio professionale, un freelance che conosci, un collega di un altro reparto, un progetto open source a cui contribuisci davvero. Non serve che paghi. Serve che ci sia qualcuno che si lamenta quando non funziona.
 
**Va agganciato al Progetto 3 o al Progetto 4.** Inizia a cercarlo durante la Fase 2, non quando ti serve.
 
---
 
## Fase 1 — Fondamenta di machine learning (settimane 1–6)
 
*Ridotta da 10 a 6 settimane: ambienti, testing e pandas li assorbi in giorni.*
 
**Obiettivo:** imparare a pensare in termini di dati, modelli e valutazione. Non padronanza degli algoritmi.
 
Cosa studiare:
 
- Python per il ML: numpy, pandas, gestione ambienti con uv, basi di pytest *(velocissimo per te)*
- ML classico con scikit-learn: regressione, classificazione, alberi e gradient boosting (XGBoost). Livello: **so usarlo e so quando serve**, non so ottimizzarlo
- I concetti irrinunciabili — questa è la parte che conta davvero: split train/validation/test, **data leakage**, overfitting e regolarizzazione, cross-validation, metriche (precision, recall, F1, ROC-AUC) e **come si sceglie la metrica giusta per un problema**
- **Experiment tracking fin da subito**: MLflow o Weights & Biases. Introdotto qui perché senza di esso "risultati misurati e documentati" resta un'aspirazione, in questa fase e in tutte le successive
- Matematica on demand: algebra lineare e probabilità solo quando servono, con i video di 3Blue1Brown
 
**Risorsa principale:** Machine Learning Specialization di Andrew Ng (Coursera, audit gratuito) oppure Kaggle Learn se preferisci il taglio pratico. **Scegline una.**
 
**Progetto 1:** pipeline ML end-to-end su un dataset reale (churn, prezzi, quello che vuoi): esplorazione → modello → valutazione onesta → breve report delle scelte. Esperimenti tracciati.
 
**Articolo 1:** "Come ho scelto la metrica di valutazione per X e perché l'accuracy era la scelta sbagliata."
 
**Milestone:** sai spiegare a voce, senza appunti, overfitting, cross-validation, leakage e criterio di scelta di una metrica. Progetto su GitHub.
 
---
 
## Fase 2 — Deep learning, transformer e open-weight (settimane 7–16)
 
**Obiettivo:** capire cosa c'è dentro un LLM per non usarlo come scatola nera, e **saper prendere un modello open e specializzarlo**. È la seconda metà che ti differenzia sul mercato europeo.
 
Cosa studiare:
 
- PyTorch: tensori, autograd, un training loop scritto a mano
- **"Neural Networks: Zero to Hero" di Karpathy**, fino a nanoGPT. Da fare **scrivendo il codice**, non guardando i video (→ tecnica della riscrittura a memoria nel documento sul metodo)
- Architettura transformer: tokenizzazione, embeddings, attention, KV cache (almeno concettualmente)
- Ecosistema Hugging Face: transformers, datasets; fine-tuning efficiente con LoRA/QLoRA
- **Piega open-weight**: come si sceglie un modello open per un caso d'uso, come si valuta (benchmark vs valutazione sul proprio dominio), quantizzazione di base (GGUF, AWQ, bitsandbytes), cosa cambia tra 7B e 70B in termini di hardware e costo
 
**Risorsa principale:** Karpathy su YouTube (gratuito) + LLM Course di Hugging Face.
 
**Progetto 2:** fine-tuning di un modello open su un compito specifico in un dominio che conosci, con **confronto documentato contro un baseline** (modello base + prompting, e possibilmente un'API commerciale) su costo, latenza e qualità. Demo su Hugging Face Spaces.
 
Il confronto è la parte che vale: chiunque sa lanciare un LoRA, quasi nessuno sa dire se ne valeva la pena.
 
**Articolo 2:** "Fine-tuning vs prompting su [dominio]: numeri, costi e quando non conviene."
 
**Milestone:** hai addestrato un mini-GPT da zero e completato un fine-tuning con risultati misurati contro un baseline.
 
> **Onestà sul livello:** questa fase dà alfabetizzazione, non competenza di training. Un LoRA singolo sul CV non è "esperienza di fine-tuning" e in colloquio la differenza emerge in due domande. Sai *usare* il fine-tuning. È già molto più della media.
 
---
 
## Fase 3 — LLM engineering (settimane 17–38) · il cuore del percorso
 
*Espansa da 12 a 22 settimane. Qui va tutto il tempo risparmiato altrove. Circa il 45% del monte ore totale.*
 
**Obiettivo:** costruire applicazioni LLM affidabili, non demo. È qui che si gioca l'occupabilità.
 
### 3a — Fondamenta applicative (settimane 17–21)
 
- API dei modelli: structured outputs, function calling, streaming, gestione errori, costi, caching
- **asyncio e concorrenza in Python.** Le app LLM sono I/O-bound per definizione: batching di richieste, rate limiting, retry con backoff, timeout, gestione del fan-out. Non saperlo si nota subito
- Prompt engineering sistematico: versioning dei prompt e test automatici, non trucchetti
- **Multimodale di base**: passare immagini e audio, OCR via modello, quando conviene rispetto a una pipeline classica
 
### 3b — Data engineering e ingestione (settimane 22–27) · *sezione nuova*
 
Il buco più grosso di ogni percorso in circolazione. **Il RAG non fallisce sul chunking: fallisce sull'ingestione.** È dove va il 60% del tempo di un progetto reale ed è quello che ti chiedono in colloquio quando vogliono capire se hai costruito qualcosa di vero.
 
- Parsing di documenti reali e sporchi: PDF con layout a colonne, tabelle, scansioni, header/footer ripetuti, documenti Office, email
- OCR quando serve, e come capire quando serve
- Normalizzazione, deduplicazione, gestione delle versioni di un documento
- **Aggiornamento incrementale dell'indice**: cosa succede quando un documento cambia, come non rifare tutto
- **Permessi per documento**: filtrare i risultati in base a chi sta facendo la domanda. Requisito di fatto in qualsiasi contesto aziendale, quasi mai trattato nei corsi
- Osservabilità della pipeline di ingestione: cosa è entrato, cosa è stato scartato e perché
 
### 3c — RAG e retrieval (settimane 28–31)
 
- Chunking (strategie e trade-off), embeddings, scelta del modello di embedding
- **Postgres sul serio**, non solo pgvector: SQL, indici, filtri su metadati combinati con la ricerca vettoriale, query plan. In alternativa Qdrant, ma Postgres è quello che troverai in azienda
- Hybrid search (BM25 + vettoriale), reranking, e **quando il reranking non serve**
- Valutazione del retrieval separata dalla valutazione della generazione
 
### 3d — Agenti (settimane 32–34)
 
- Tool use, loop di controllo, gestione dello stato, MCP (Model Context Protocol)
- Un framework a scelta: LangGraph o gli SDK nativi dei provider
- **Quando un agente NON serve** — che è la maggior parte delle volte, e saperlo dire è un segnale senior
- **Sicurezza degli agenti**, oltre la prompt injection: design dei tool a privilegio minimo, sandboxing dell'esecuzione, controllo dell'egress, punti di approvazione umana. Con gli agenti in produzione questo è *il* tema del momento
- Guardrails, gestione dei dati personali, prompt injection
 
### 3e — Evals (settimane 35–38) · *il vero differenziatore*
 
- Costruzione di un dataset di test a partire da casi reali
- LLM-as-judge: come si costruisce, come si valida contro giudizi umani, dove sbaglia
- Regression testing dei prompt (promptfoo, Langfuse, LangSmith)
- Metriche separate per retrieval e generazione
- **Eval online**: cosa si misura in produzione e come si raccoglie il feedback degli utenti
 
**Risorsa principale:** short courses di DeepLearning.AI (RAG, agenti, evals) + il libro *AI Engineering* di Chip Huyen + i cookbook di Anthropic e OpenAI.
 
**Progetto 3:** applicazione completa RAG + agente su un dominio concreto, **con ingestione di documenti reali e sporchi** (non un corpus pulito scaricato da Kaggle), suite di eval e demo pubblica. **Candidato principale per il requisito dell'utente vero.**
 
**Articolo 3:** "Cosa si rompe davvero in un RAG in produzione" — quello che ti farà leggere dai recruiter tecnici.
 
**Milestone:** la tua app passa una suite di eval con metriche documentate nel README, su documenti che nessuno ha ripulito per te.
 
> **Da qui in avanti la Fase 5 gira in parallelo.** Settimana 32 ≈ mese 8: si comincia a candidarsi.
 
---
 
## Fase 4 — Produzione, LLMOps e serving open (settimane 39–46)
 
*Comprimibile a 5 settimane se i tuoi 5 anni sono stati di backend con Docker, CI/CD e cloud. Da tenere intera se sei stato prevalentemente frontend: in quel caso è il pezzo che ti differenzia di più.*
 
**Obiettivo:** deployare come si fa in azienda. Pochi candidati sanno farlo davvero.
 
Cosa studiare:
 
- FastAPI per esporre il servizio, Docker per containerizzarlo
- **Streaming verso il frontend (SSE), job in background, code, retry e timeout end-to-end.** Vantaggio competitivo diretto dei tuoi 5 anni di web: sai già cosa significa un'interfaccia che deve reggere una risposta lenta
- CI/CD con GitHub Actions
- Un cloud a scelta: **Azure** (molto richiesto nelle enterprise europee) o AWS (il più diffuso in assoluto) — deploy di container, secrets, controllo dei costi
- Observability: logging, tracing delle chiamate LLM, monitoring di qualità e costi (Langfuse)
- **Serving di modelli open con vLLM**: batching continuo, KV cache, quantizzazione in serving, e soprattutto **il calcolo economico** — a che volume conviene self-hosting rispetto alle API. Questa è la competenza che ti rende utile alle aziende europee che devono tenere i dati in casa, ed è il tuo ponte verso il profilo 2
 
**Progetto 4:** il Progetto 3 portato in produzione vera: CI/CD, monitoring, gestione degli errori, dashboard dei costi. Se non l'hai già fatto in Fase 3, **è qui che deve arrivare l'utente vero**.
 
**Articolo 4:** "Self-hosting di un modello open vs API: il break-even reale, con i numeri."
 
**Milestone:** l'app è live, con pipeline di deploy automatica e dashboard di monitoring.
 
---
 
## Fase 5 — Mercato (dalla settimana 32, in parallelo)
 
**Non è una fase finale. Comincia a mese 8, mentre la Fase 3 è ancora in corso.** I colloqui sono la diagnostica migliore che esista su cosa ti manca: farli tardi significa scoprire tardi.
 
- **GitHub curato**: 3–4 repository con README che spiegano problema, architettura, decisioni e demo. Le *decisioni* sono la parte che leggono
- **5 articoli tecnici**, uno per fase, scritti *durante* e non alla fine
- CV e LinkedIn con le parole chiave che i recruiter cercano: RAG, agents, evals, LLMOps, fine-tuning, vLLM, open-weight
- **Posizionamento**: presentati come *software engineer con 5 anni di esperienza che è andato sull'AI*, non come uno che sta cambiando mestiere. È un vantaggio enorme e va detto per primo
- Preparazione colloqui: live coding in Python + **system design di sistemi LLM** (ti chiederanno di progettare un RAG o un agente alla lavagna). Fai simulazioni: l'AI è un ottimo intervistatore
- **EU AI Act**: categorie di rischio, obblighi per i sistemi. Pochissimi candidati ce l'hanno e nelle aziende regolamentate pesa. Bastano 4–5 ore
- Certificazione opzionale: Azure AI Engineer (AI-102). Utile per le enterprise, non sostituisce mai il portfolio
 
**Milestone finale:** 3–4 progetti pubblici di cui almeno uno con utenti veri, 5 articoli, candidature attive da mesi.
 
---
 
## Se arrivano opportunità prima
 
Se dopo la Fase 3 ricevi proposte concrete, **accettale e completa la Fase 4 lavorando**. Il mercato premia chi costruisce, non chi completa programmi. Un anno di lavoro vale più della metà finale di qualsiasi percorso di studio.
 
---
 
## Dopo i 12 mesi: il bivio
 
A mese 18–24, da dentro un lavoro e con accesso a GPU e problemi veri, scegli.
 
**Senior applicativo:** evals come *sistema* e non come suite — offline e online, A/B test in produzione, annotazione umana, drift detection, ciclo di feedback dagli utenti. Più: modellazione dei costi su volumi veri, gestione di incidenti, e il giudizio su **quando non usare un LLM**. Quest'ultima è la cosa che più distingue un senior in colloquio.
 
**Model engineer** (l'obiettivo dichiarato): post-training oltre l'SFT (DPO e affini), training distribuito, quantizzazione avanzata, ottimizzazione dell'inferenza (KV cache, batching continuo, speculative decoding, cosa fa vLLM sotto il cofano), e l'abitudine a leggere e reimplementare paper. È un'altra fase intera da 300+ ore. **La porta d'ingresso è l'inference/training infrastructure, non la ricerca.**
 
**Platform / LLMOps:** Kubernetes, scheduling di GPU, autoscaling di workload GPU, serving multi-tenant, model gateway, FinOps dell'inferenza.
 
Trasversalmente, il vero salto di livello non è tecnico: è saper decidere cosa costruire e cosa no, e saperlo argomentare.
 
---
 
## Riepilogo risorse (quasi tutte gratuite)
 
| Fase | Risorsa principale |
|---|---|
| 1 | Machine Learning Specialization — Andrew Ng (Coursera, audit gratuito) *oppure* Kaggle Learn |
| 2 | Neural Networks: Zero to Hero — Karpathy (YouTube) + LLM Course di Hugging Face |
| 3 | Short courses DeepLearning.AI (RAG, agenti, evals) + *AI Engineering* di Chip Huyen + cookbook Anthropic e OpenAI |
| 4 | Documentazione FastAPI, vLLM, Langfuse + docs del cloud scelto |
| 5 | Testo dell'EU AI Act (sezioni su categorie di rischio) |
| Trasversale | 3Blue1Brown per la matematica on demand |
 
*AI Engineering* di Chip Huyen è l'unico acquisto davvero consigliato.
 
---
 
## Riepilogo del calendario
 
| Settimane | Fase | Deliverable |
|---|---|---|
| 1–6 | Fondamenta ML | Progetto 1 + Articolo 1 |
| 7–16 | Deep learning + open-weight | Progetto 2 + Articolo 2 |
| 17–38 | LLM engineering | Progetto 3 + Articolo 3 |
| 32 → | **Candidature attive** | CV, LinkedIn, simulazioni |
| 39–46 | Produzione e LLMOps | Progetto 4 + Articolo 4 |
| 47–48 | Consolidamento | Articolo 5, chiusura portfolio |