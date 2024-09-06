# Reference for sqlite db usage:
# https://mlflow.org/docs/latest/tracking/tutorials/local-database.html#set-the-tracking-uri-to-a-local-sqlite-database
MLFLOW_TRACKING_URI=sqlite:///mlruns.db cd artifacts && poetry run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db