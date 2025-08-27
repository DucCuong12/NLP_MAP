import transformers
from typing import Any, Dict, List
from transformers import AutoTokenizer,DataCollatorWithPadding
from dataclasses import dataclass
from tokenization import TokenizeData
import numpy as np


@dataclass
class RankerDataCollator(DataCollatorWithPadding):
    tokenizer: AutoTokenizer
    negative_dataset : Any
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        batch_size = len(features)
        
        if batch_size <= 1:
            features = {k:v for k,v in feature.items() if k != 'row_id'}
            return self.tokenizer.pad(features, padding=True, return_tensors="pt")

        
        
        positive_features = list(features)
        negative_indices = []
        
        for feature in positive_features:
            idx = feature["row_id"]
            offset= idx*73
            negative_list = np.random.choice(range(offset+1,offset+73),size=batch_size,replace=False)
            negative_list = list(negative_list)
            negative_indices += negative_list
    
        
        # negative_indices = [(i + np.random.randint(1, batch_size)) % batch_size for i in range(batch_size)]
        negative_features = [self.negative_dataset[i] for i in negative_indices]
        
        all_features = positive_features + negative_features
        final_features = []

        for feature in all_features:
            clean = {k:v for k,v in feature.items() if k != 'row_id'}
            final_features.append(clean)

        
        padded_batch = self.tokenizer.pad(
            final_features,
            padding=True,
            return_tensors="pt",
        )
        return padded_batch