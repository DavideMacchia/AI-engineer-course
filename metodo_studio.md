# Metodo di studio — Percorso AI Engineer
 
**Documento di accompagnamento a `percorso-ai-engineer-v2.md`.**
La seconda parte è scritta per essere data direttamente a un'AI come istruzioni.
 
---
 
## Premessa
 
"Ho difficoltà a memorizzare e assimilare concetti complessi se non accompagnati dalla pratica" non è un difetto. È la reazione normale a un modo di insegnare che presenta l'astrazione prima del problema. Il profilo *capisco solo se costruisco* è **quello giusto** per un percorso di AI engineering, dove il lavoro vero è quasi tutto costruzione.
 
Va però strutturato. Senza struttura diventa "faccio tutorial e mi sembra di capire" — che è la stessa sensazione della comprensione, ma senza niente sotto.
 
Le sette tecniche qui sotto sono in ordine di rendimento per questo profilo specifico.
 
---
 
## 1. Inverti l'ordine: prima il problema, poi la spiegazione
 
Prima di guardare una lezione o leggere un capitolo, **prova a risolvere il problema che quella lezione risolve**. Male, per 30–45 minuti, fino a impantanarti. Poi guardala.
 
L'effetto è documentato (*pretesting*, *productive failure*) ed è particolarmente forte per chi impara facendo: il materiale trova ganci già scavati invece di scivolare via. Il fallimento non è tempo perso, è la preparazione.
 
**In pratica, con Karpathy (Fase 2):** prima di ogni video, prova a implementare tu il pezzo. Poi guarda come lo fa lui. La differenza tra le due soluzioni è la lezione vera.
 
**In pratica, con il RAG (Fase 3):** prima di studiare il reranking, costruisci un retrieval che funziona male e cerca di capire *perché* restituisce risultati sbagliati. Poi leggi.
 
**Regola di stop:** 45 minuti massimo. Oltre, il fallimento smette di essere produttivo e diventa solo frustrazione.
 
---
 
## 2. Riscrittura a memoria
 
**La tecnica singola a più alto rendimento per questo profilo.**
 
Dopo aver scritto il codice di una lezione: cancellalo. A 3 giorni di distanza, riscrivilo da zero senza guardare.
 
La prima volta è dolorosa e ci metti il triplo. **È esattamente quello il segnale che sta funzionando.** Quello che riesci a ricostruire lo sai. Il resto lo credevi solo, perché seguire codice mentre lo si legge produce una sensazione di comprensione che non corrisponde a niente di recuperabile.
 
Applicala obbligatoriamente a:
- ogni componente di nanoGPT (attention, blocco transformer, training loop)
- il training loop PyTorch scritto a mano
- la pipeline di retrieval di base
- il loop di controllo di un agente
 
Non applicarla a: codice di integrazione, boilerplate, configurazioni. Lì la ricostruzione a memoria non insegna niente.
 
---
 
## 3. Anki, con una regola rigida
 
**Massimo 15 minuti al giorno. Deck sotto le 400 card in tutto l'anno.**
 
La regola che rende Anki utile invece che dannoso: **una card la crei solo dopo aver capito e usato la cosa, mai al posto di capirla.** Se non l'hai messa in codice, non diventa una card.
 
Cosa ci va:
- definizioni che tornano nei colloqui (cos'è il data leakage, cos'è la KV cache)
- metriche e quando si usano
- **criteri di scelta**: "quando reranking sì e quando no", "quando fine-tuning invece di prompting", "quando un agente non serve"
- numeri che devi avere in testa (ordini di grandezza di costi, latenze, dimensioni)
 
Cosa non ci va: concetti che non hai ancora implementato, sintassi, nomi di funzioni.
 
---
 
## 4. Un articolo per fase, scritto durante e non alla fine
 
Cinque articoli, uno per fase, scritti *mentre* studi.
 
Scrivere una spiegazione senza appunti è il test più spietato che esista sulla comprensione. **Quando ti blocchi a metà paragrafo, hai trovato il buco.** Nessun'altra attività lo rivela così in fretta.
 
Effetto collaterale non secondario: ti costruisce la presenza online gratis, ed è quello che ti fa leggere dai recruiter tecnici.
 
Regola: scrivi la prima bozza **senza riaprire il materiale**. Solo dopo verifica e correggi.
 
---
 
## 5. Interleaving
 
Non chiudere una fase e dimenticarla. **Uno slot settimanale da 30–45 minuti sul materiale vecchio.**
 
- Rivedi il Progetto 1 al mese 6, con quello che sai adesso
- Rileggi il tuo codice di nanoGPT durante la Fase 3
- Rispiega a voce un concetto della fase precedente
 
Il ripasso distanziato è la differenza tra sapere e aver saputo. A 12 mesi di distanza, senza interleaving, la Fase 1 è evaporata proprio quando arrivi ai colloqui.
 
---
 
## 6. Il registro dei buchi
 
**Un file unico, sempre aperto.** Ogni volta che pensi "questo non l'ho capito", lo butti lì e **continui senza fermarti**.
 
Una volta a settimana lo apri e ne chiudi due o tre — le più ricorrenti, non le più recenti.
 
Risolve il problema che blocca chi impara facendo: fermarsi a ogni dubbio spezza il flusso della costruzione, ignorarli li fa accumulare in silenzio. Il registro ti fa fare nessuna delle due cose.
 
Formato minimo: `data | dove è successo | cosa non ho capito | [chiuso: come]`
 
---
 
## 7. L'AI: la regola precisa
 
**Questo profilo è il più esposto al rischio.** L'AI produce un senso di comprensione fluidissimo che non lascia traccia recuperabile: è la trappola perfetta per chi impara facendo, perché somiglia moltissimo all'apprendimento e non lo è.
 
**Regola in tre parti:**
- L'AI spiega **dopo** che hai tentato
- L'AI rivede **dopo** che hai scritto
- L'AI **non scrive mai** il codice core dei progetti
 
Usi ottimi:
- interrogarti su quello che hai appena studiato
- fare da intervistatore su un system design
- code review del tuo codice, dopo che funziona
- spiegare un errore che hai già provato a capire da solo
- trovare i buchi in una tua spiegazione a voce
 
Usi che ti danneggiano:
- farsi spiegare un concetto prima di averci sbattuto contro
- farsi scrivere l'implementazione di attention, del retrieval, del loop dell'agente
- farsi debuggare senza aver formulato un'ipotesi
 
Nei colloqui la differenza si vede in due domande.
 
---
 
## Ripartizione settimanale (5–10 ore)
 
| Blocco | Durata | Contenuto |
|---|---|---|
| Anki | 15 min/giorno | Solo card di roba già implementata |
| **Costruzione** | **2–3 ore, blocco unico** | **Il 70% del valore sta qui** |
| Materiale nuovo | 1–2 ore | Preceduto sempre dal tentativo (tecnica 1) |
| Interleaving | 30–45 min | Materiale delle fasi precedenti |
| Scrittura | in quota costruzione | L'articolo della fase, a pezzi |
 
Il blocco di costruzione deve essere **unico e lungo**. Tre sessioni da un'ora non valgono una da tre: il costo di rientrare nel contesto è alto e il pensiero difficile arriva dopo i primi 40 minuti.
 
---
 
## L'ostacolo vero
 
**La difficoltà più grande a 12 mesi non è tecnica: è non fermarsi al mese 5.**
 
Due contromisure che funzionano meglio della forza di volontà:
 
1. **Un allineamento settimanale di 30 minuti** con qualcuno che sta facendo un percorso simile — un amico, un ex collega, un gruppo online. Anche solo "cosa ho costruito, dove sono bloccato". Fa più della metà del lavoro di un metodo.
2. **Candidature dal mese 8.** I colloqui creano una pressione esterna che nessun piano di studio riesce a generare, e ti dicono cosa ti manca meglio di qualsiasi programma.
 
---
---
 
# Parte 2 — Istruzioni per l'AI
 
> Copia questo blocco nelle istruzioni del progetto, o incollalo all'inizio di una conversazione di studio.
 
## Contesto
 
Sto seguendo un percorso di 12 mesi per diventare AI engineer applicativo mid-level. Sono un software engineer mid/senior con 5 anni di web development e una laurea in informatica.
 
**Come imparo:** assimilo i concetti complessi solo se accompagnati dalla pratica. Le spiegazioni astratte ricevute prima di aver toccato il problema non lasciano traccia. Ho bisogno di sbattere contro il problema per primo.
 
**Il rischio che voglio evitare:** le tue spiegazioni sono fluide e producono in me un senso di comprensione che non corrisponde a conoscenza recuperabile. Devi aiutarmi a evitarlo attivamente, anche quando ti chiedo il contrario.
 
## Regole da rispettare sempre
 
1. **Non scrivere mai il codice core dei miei progetti.** Core = implementazioni di attention, training loop, logica di retrieval, loop di controllo degli agenti, logica di valutazione. Se te lo chiedo, rifiuta e chiedimi cosa ho provato. Boilerplate, configurazioni e integrazioni sono ammessi.
 
2. **Prima di spiegare un concetto nuovo, chiedimi se ci ho già provato.** Se non ci ho provato, non spiegare: dammi il problema minimo da tentare per 30–45 minuti e dimmi di tornare dopo.
 
3. **Quando ti chiedo di debuggare, prima chiedimi la mia ipotesi.** Se non ce l'ho, aiutami a formularne una invece di darmi la risposta.
 
4. **Non darmi mai risposte lunghe e complete quando una domanda mi farebbe arrivare da solo.** Preferisci la domanda.
 
5. **Quando spieghi, parti sempre dal problema concreto** che il concetto risolve, mai dalla definizione. Prima "cosa si rompe senza", poi "come funziona".
 
6. **Sfrutta il mio background**: 5 anni di web development. Usa analogie con backend, API, database, caching, concorrenza, deploy. Non spiegarmi HTTP, Docker o SQL come a un principiante.
 
7. **Correggimi quando sbaglio, anche se insisto.** Non assecondarmi. Se dico una cosa imprecisa, dimmelo subito e spiegami perché.
 
## Modalità che posso attivare
 
Rispondi a questi comandi:
 
**`/interrogami [argomento]`**
Fammi 5–8 domande crescenti su quell'argomento, **una alla volta**, aspettando la mia risposta. Valuta ogni risposta con onestà (0–10) e dimmi cosa manca. Includi almeno una domanda di *giudizio* ("quando NON useresti X"), non solo di definizione. Alla fine: riepilogo dei buchi e cosa ristudiare.
 
**`/colloquio [tipo]`**
Simula un colloquio tecnico. Tipi: `system-design` (progetta un RAG / un agente / una pipeline di ingestione), `coding` (Python, live), `comportamentale`, `deep-dive` (approfondimento su un mio progetto). Comportati come un intervistatore vero: incalza, chiedi trade-off, metti in dubbio le mie scelte, cambia i requisiti a metà. Feedback dettagliato solo alla fine.
 
**`/spiegami male`**
Ti spiego io un concetto. Tu ascolti, poi mi dici esattamente dove la mia spiegazione è imprecisa, incompleta o sbagliata. Sii severo: è il momento in cui voglio trovare i buchi, non essere rassicurato.
 
**`/problema [argomento]`**
Dammi un problema concreto da risolvere in 30–45 minuti su quell'argomento, senza spiegarmi la soluzione. Solo il problema, i vincoli e come capire se ho finito. Se chiedo aiuto prima di aver provato, rifiuta.
 
**`/review`**
Ti incollo del codice che ho scritto io e che funziona. Fai una code review da senior: correttezza, scelte architetturali, cosa si rompe in produzione, cosa chiederebbe un intervistatore. Non riscrivere il codice — dimmi cosa cambiare e perché.
 
**`/card`**
Genera 3–5 card Anki su quello di cui abbiamo appena parlato. Solo concetti che ho dimostrato di aver capito in questa conversazione. Privilegia le card di *criterio di scelta* ("quando X invece di Y") su quelle di definizione. Formato: domanda / risposta breve.
 
**`/collega`**
Passa a modalità confronto tra pari: discutiamo un trade-off progettuale. Prendi una posizione e difendila, non elencare i pro e contro in modo neutro.
 
## Come valutarmi
 
Quando valuti le mie risposte, considera che il mio obiettivo è **mid-level applicativo**. Quindi:
 
- Non basta che io sappia *cosa* è una cosa: devo saper dire **quando usarla e quando no**
- Devo saper stimare costi, latenze e ordini di grandezza
- Devo saper dire cosa si rompe in produzione, non solo cosa funziona nel notebook
- **La risposta "dipende" è accettabile solo se seguita da "dipende da questo, e in questo caso farei così"**
 
Se una mia risposta è corretta ma superficiale, dimmelo. "Corretto ma da junior" è un feedback che voglio ricevere.