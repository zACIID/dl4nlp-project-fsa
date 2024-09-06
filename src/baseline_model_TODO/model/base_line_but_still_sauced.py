from enum import Enum
from sklearn.svm import SVR
import torch.nn as nn
from src.fine_tuned_finbert.models.fine_tuned_finbert import PRE_TRAINED_MODEL_PATH
from transformers import (
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Union
from numpy import ndarray
import pandas as pd


class BLBSSType(Enum):
    FinBERT = 0
    SVM = 1


class BLBSSModel(nn.Module):
    def __init__(
            self, model_type: BLBSSType, model_path: str = PRE_TRAINED_MODEL_PATH,
            SVR_dataset: pd.DataFrame = None, SVR_labels: ndarray[float] = None
    ):
        super().__init__()

        self._model_type: BLBSSType = model_type

        if self._model_type == BLBSSType.FinBERT:
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self._model = SVR()
            print("A pier non piacciono i costruttori con side effect, ma d'altronde cosa mai potrebbe andare storto")
            self._model.fit(SVR_dataset, SVR_labels)

    def forward(self, **inputs) -> Union[SequenceClassifierOutput, ndarray]:
        if self._model_type == BLBSSType.FinBERT:
            return self._model(**inputs)
        else:
            return self._model.predict(**inputs)


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
\_ |__    ____                                                      
 | __ \ _/ __ \                                                     
 | \_\ \\  ___/                                                     
 |___  / \___  >                                                    
     \/      \/                                                     
        .__ .__               .___                                  
  ____  |__||  |    ____    __| _/  __ __ ______                    
 /  _ \ |  ||  |  _/ __ \  / __ |  |  |  \\____ \                   
(  <_> )|  ||  |__\  ___/ / /_/ |  |  |  /|  |_> >                  
 \____/ |__||____/ \___  >\____ |  |____/ |   __/                   
                       \/      \/         |__|          
"""
