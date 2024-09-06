# TODO cose da fare:
# - vedi todo nel modello
# - implementare hyperopt e training (unico parametro da scegliere sembra n_layers e forse out_layers, per questo sento biagio prima
#   - devo aggiungere entry point su MLProject
#   - a sto punto completo il refactoring: aggiungo il suffisso del modello a tutti gli entry point specifici
# - implementare evaluation -> questo lascerei a ruie e altri anche perche' ci stanno lavroando loro al momento
# - provare a runnare... per questo dovrei forse sistemare la cosa dei nan nel preprocessing, metto un cerotto nelle collate_fn e rimpiazzo zero con nan????