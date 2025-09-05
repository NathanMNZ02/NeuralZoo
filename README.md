# NeuralZoo

NeuralZoo è un repository che raccoglie diverse implementazioni di **reti neurali** per differenti scopi. L'obiettivo è fornire una collezione organizzata di modelli, dataset e script di addestramento/utilizzo.

---

## Struttura del repository

### ModelZoo
Ogni sotto-cartella contiene differenti implementazioni di una specifica tipologia di rete neurale, insieme alle classi necessarie per l'addestramento e l'inferenza.

### DataSetZoo
Ogni sotto-cartella contiene la classe per implementare ed introdurre nel codice un determinato dataset ed eventuali funzioni per il preprocessing:
- svhn, dataset rappresentante numeri civici, scaricabile dalla pagina: http://ufldl.stanford.edu/housenumbers/

Puoi installare tutte le dipendenze con:

```bash
pip install -r requirements.txt
