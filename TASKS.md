# Subdivision of the project into tasks

1. Preprocessing
   1. in sostanza bisogna partire dai due dataset citati nell'excalidraw (o nel documento del progetto)
2. Training the two models
  1. FinBERT finetuning
  2. Training the hand-engineered features based MLP
  3. IDEE:
     1. l'ultimo layer FC dei due modelli ha la stessa dimensionalita', in modo che quando poi uniamo i due modelli inizializzando il layer che gli unisce come la media dei loro ultimi layer
  4. DOMANDE:
     1. che loss usiamo? MSE dato che e' regressione? Magari ci rileggiamo i paper di quelli che hanno fatto la challenge per capire che loss hanno usato
  5. TODO: per Biagio
     1. come implementare LoRA su PyTorch (Lightning)
     2. implementare con PyTorch Lightning il modello da fine tunare, esempio utile: https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py
     3. scriptino che prende in input il percorso dei file di serializzazione di uno o piu' modelli e li carica con PyTorch Lightning
        1. questo e' utile perche poi quando abbiamo finito l'allenamento dei nostri modelli procediamo con quello migliore
        2. deve essere uno script python che va lanciato da riga di comando
3. Training the Ensemble End-to-End
4. Analysis (TODO nel nostro paper del progetto abbiamo scritto che)
  1. Baseline models implementation
     1. solo finbert
     2. modello 1 (finbert finetunato)
     3. modello 2 (MLP hand-engineered based)
     4. cos'altro avevamo scritto nel documento del progetto da usare come paragone?
  2. Analisi con varie metriche TODO
     1. TODO elencare quali sono le metriche. Ricordo che la metrica ufficiale della challenge era la cosine similarity
     2. confrontare se possibile numero di parameteri della nostra rete rispetto a quelle delle challenge
5. Report del progetto (TODO com'e' fatto?)

```python
class WithClassificationLayers
  def __init__(self, base: nn.Module, classification_layers: List[nn.Module])
    pass
```