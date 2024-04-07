../../.venv/bin/jupytext --set-formats ipynb,py:percent "$1".ipynb &&\
../../.venv/bin/jupytext --sync "$1".py