# TODO ( ͡° ͜ʖ ͡°) implement, take src/fine_tuned_finbert/evaluation/evaluate_finbert as example It is interesting
#  for us to evaluate every component of our ensemble so that we can see if we actually improved stuff at the end

# TODO: see their cosine similarity - https://alt.qcri.org/semeval2017/task5/index.php?id=evaluation

# TODO: Our primary evaluation metric for financial sentiment analysis is the weighted cosine similarity,aligning
#  with the SemEval2017 challenge’s official evaluation method. This metric measures the proximity between predicted
#  sentiment scores and the gold standard. Additional standard metrics, including precision, recall, and F1 score

# TODO unire dataset, dividere, preprocessing, vono venire fuori due database separati train e test se vai a vedere
#  c’e’ un flag che viene passato allo script di preprocessing di semeval che decide su quale dataset viene fatto il
#  preprocessing serve che siano due file separati perche’ poi gli script “evaluation” caricano solo il dataset di
#  test e ci sono dei datamodules solo per il dataset di test (Semeval2017Test tipo

