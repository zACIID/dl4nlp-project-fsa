import torch.nn as nn
from src.fine_tuned_finbert.models.fine_tuned_finbert import PRE_TRAINED_MODEL_PATH
from transformers import (
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput


class BLBSSModel(nn.Module):
    def __init__(self, model_path: str = PRE_TRAINED_MODEL_PATH):
        super().__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def forward(self, **inputs) -> SequenceClassifierOutput:
        return self._model(**inputs)


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
