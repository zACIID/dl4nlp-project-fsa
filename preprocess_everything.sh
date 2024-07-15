# NOTE: Spark cluster must be running
export PYTHONPATH=src:$PYTHONPATH  \
&& poetry run python src/fine_tuned_finbert/datasets/stocktwits_crypto/preprocessing.py --drop-neutral-samples \
&& poetry run python src/fine_tuned_finbert/datasets/stocktwits_crypto/preprocessing.py \
&& poetry run python src/fine_tuned_finbert/datasets/semeval_2017/preprocessing.py \
&& poetry run python src/fine_tuned_finbert/datasets/semeval_2017/preprocessing.py --get-train-dataset \
&& poetry run python src/data/hand_engineered_mlp_TODO/stocktwits_crypto/preprocessing.py --drop-neutral-samples \
&& poetry run python src/data/hand_engineered_mlp_TODO/stocktwits_crypto/preprocessing.py \
&& poetry run python src/data/hand_engineered_mlp_TODO/semeval_2017/preprocessing.py \
&& poetry run python src/data/hand_engineered_mlp_TODO/semeval_2017/preprocessing.py --get-train-dataset \
# TODO possibly other preprocessing scripts
# TODO poetrypythonsrc src/data/hand_engineered_mlp/stocktwits_crypto/preprocessing.py per runnare script di preprocessing singolarmente da terminal