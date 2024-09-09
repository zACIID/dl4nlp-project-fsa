from enum import Enum
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from sklearn.svm import SVR
from transformers import (
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.fine_tuned_finbert.models.fine_tuned_finbert import PRE_TRAINED_MODEL_PATH


class BLBSSType(Enum):
    FinBERT = 0
    SVR = 1


class BLBSSModel(nn.Module):
    def __init__(
            self,
            model_type: BLBSSType,
            model_path: str = PRE_TRAINED_MODEL_PATH,
            SVR_dataset: pd.DataFrame = None,
            SVR_labels: pd.DataFrame = None
    ):
        super().__init__()

        self._model_type: BLBSSType = model_type

        if self._model_type == BLBSSType.FinBERT:
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self._model = SVR()
            self._svr_x: pd.DataFrame = SVR_dataset
            self._svr_y: pd.DataFrame = SVR_labels

    def predict(self, finbert_input=None, svr_input=None) -> Union[SequenceClassifierOutput, ndarray]:
        if self._model_type == BLBSSType.FinBERT:
            return self._finbert_predict(finbert_input)
        else:
            return self._model.predict(**svr_input)

    def _finbert_predict(self, finbert_input):
        # _model here is finbert
        self._model.eval()  # Call this explicitly because this is external to PytorchLightning
        with torch.no_grad():
            output = self._model(**finbert_input)
            return self._to_sentiment_score(output)

    def _to_sentiment_score(self, output: SequenceClassifierOutput) -> torch.Tensor:
        # NOTE:
        # Classes are { 0: bearish, 1: neutral, 2: bullish } for the
        #   ahmedrachid/FinancialBERT-Sentiment-Analysis model
        # Classes are { 0: positive, 1: negative, 2: neutral } for the
        #   ProsusAI/finbert model

        # Transpose because it is a batch of 3-elements tensors
        probabilities = F.softmax(output.logits, dim=1).T
        # bearish_prob, bullish_prob = probabilities[0], probabilities[2] # TODO for ahmedrachid
        bearish_prob, bullish_prob = probabilities[1], probabilities[0]

        # This is also how ProsusAI/finbert predicts sentiment score:
        #   positive prob - negative prob, and then it uses MSE loss
        pred_sentiment_score = bullish_prob - bearish_prob
        return pred_sentiment_score

    def fit(self) -> None:
        if self._model_type == BLBSSType.SVR:
            self._model.fit(self._svr_x, self._svr_y)


"""
__________                      __                                  
\______   \  ____    ____      |__|                                 
 |    |  _/_/ __ \  /    \     |  |                                 
 |    |   \\  ___/ |   |  \    |  |                                 
 |______  / \___  >|___|  //\__|  |                                 
        \/      \/      \/ \______|                                 
                        ___.             __     __                  
 ___.__.  ____   __ __  \_ |__    ____ _/  |_ _/  |_   ____ _______ 
<   |  | /  _ \ |  |  \  | __ \ _/ __ \\   __\\   __\_/ __ \\_  __ \
 \___  |(  <_> )|  |  /  | \_\ \\  ___/ |  |   |  |  \  ___/ |  | \/
 / ____| \____/ |____/   |___  / \___  >|__|   |__|   \___  >|__|   
 \/                          \/      \/                   \/        
___.                                                                
\_ |__    ____      __ __ ______                                                  
 | __ \ _/ __ \    |  |  \\____ \                                                 
 | \_\ \\  ___/    |  |  /|  |_> >                                                
 |___  / \___  >   |____/ |   __/                                                 
     \/      \/           |__|                                                    
"""
