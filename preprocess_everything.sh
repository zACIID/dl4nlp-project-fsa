# NOTE: Spark cluster must be running
export PYTHONPATH=src:$PYTHONPATH  \
&& python src/fine_tuned_finbert/datasets/stocktwits_crypto/preprocessing.py --drop-neutral-samples \
&& python src/fine_tuned_finbert/datasets/stocktwits_crypto/preprocessing.py \
&& python src/fine_tuned_finbert/datasets/semeval_2017/preprocessing.py \
&& python src/fine_tuned_finbert/datasets/semeval_2017/preprocessing.py --get-train-dataset \
&& python src/hand_eng_mlp_TODO/datasets/semeval_2017/preprocessing.py \
#&& python src/data/hand_engineered_mlp/stocktwits_crypto/preprocessing.py --drop-neutral-samples \
#&& python src/data/hand_engineered_mlp/stocktwits_crypto/preprocessing.py \
#&& python src/data/hand_engineered_mlp/semeval_2017/preprocessing.py \
#&& python src/data/hand_engineered_mlp/semeval_2017/preprocessing.py --get-train-dataset \
# TODO possibly other preprocessing scripts
# TODO poetrypythonsrc src/data/hand_engineered_mlp/stocktwits_crypto/preprocessing.py per runnare script di preprocessing singolarmente da terminal