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

- **Datasets**:
  - exploratory data analysis **TASK 1 -> SARA** -> *there are a lot of pre-made libraries that we could leverage about EDA*
      - check this: https://freedium.cfd/https://medium.com/geekculture/10-automated-eda-libraries-at-one-place-ea5d4c162bbb
    - Make a couple of plots about dataset
      - I know there are some libraries somewhere that give a great exploratory analysis of the dataset
          - missingno was one? I don't remember, but use them and then ChatGPT for the plots.
      - what I want to know is
          - the number of text samples per label
          - some examples for each class
          - the maximum length in words and characters of each tweet
  - implement preprocessing with and without the neutral label **PIER**
  - implement the whole preprocessing for the hand-eng MLP **TASK 2 (HUGE) -> BIAGIO + BENJAMIN -> DOUBLE B**
    - what features to use inside this model?
      - sentence embeddings: ELMo/BERT embeddings or Glove? (dynamic vs static?) dynamic probably better **TASK 2.1**
      - senticnet scores (sarcasm, what else?)? make requests to the senticnet api and extract specific scores for each sentence/tweet **TASK2.2**
          - connect and fecth shit from the API **TASK 2.2.1** 
          - hand-engineered features: **TASK 2.2.2**
            - ratio of positive and negative words
            - number of emojis?
            - something else??
    - *at the end of the day we need to have (1) a preprocessing script like stocktwits_crypto.py and (2) a data module to use in the trainer*
  - implement the test dataset: (1) a script to preprocess it and (2) a data module **TASK 3**
    - https://bitbucket.org/ssix-project/semeval-2017-task-5-subtask-1/src/beadeb1fd0f9b8093e4828a198a92e651a4e10c6/Microblogs_Testdata.json?at=master&fileviewer=file-view-default
  - implement data module for end-to-end model

- **Model**:
  - implement the hand-eng MLP **TASK 4 -> BIAGIO**
  - implement the end-to-end model **TASK 5**

- **OTHER stuff**
  - ...