# Datasets Notes

## Training and Validation Datasets

- Elkulako datasets from HuggingFace are weird:
  - the stocktwits-emoji one has no labels
  - the stocktwits-crypto one is an excel that has labels (0 neutral, 1 bearish, 2 bullish) that I have to manually download and parse
- This other [dataset](https://www.kaggle.com/datasets/frankcaoyun/stocktwits-2020-2022-raw) from Kaggle is a bunch of excel files that should be merged and then processed a bit.
  - try to do this using spark to speed up stuff, leverage multiple cores and compress the final dataset into parquet which takes less space
  - I have to extract these two fields:
    - the text field
    - the entity field contains a dictionary of stuff, inside which is also a sentiment attribute that I have to extract to turn into label
- I might want to reelaborate the labels of both dataset into -1 for bearish, 0 for neutral and 1 for bullish
- Make a couple of plots about dataset 
  - I know there are some libraries somewhere that give a great exploratory analysis of the dataset
    - missingno was one? I don't remember, but use them and then ChatGPT for the plots.
  - what I want to know is 
    - the number of text samples per label 
    - some examples for each class
    - the maximum length in words and characters of each tweet

- *Do I also want to pre-split the data into train-val-test? I think I'll do it on the fly in the datamodule*

## Test Dataset

This is the one from the SemEval challenge: https://bitbucket.org/ssix-project/semeval-2017-task-5-subtask-1/src/master/