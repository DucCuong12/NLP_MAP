from typing import Any, Dict, List
from transformers import AutoTokenizer,DataCollatorWithPadding
from dataclasses import dataclass

import torch


@dataclass
class RankerDataCollator(DataCollatorWithPadding):
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature['label'] for feature in features]
        clean_features = []
        for feature in features:
            clean = {k: v for k, v in feature.items() if k not in ['row_id','label']}
            clean_features.append(clean)
        
        
        clean_features = self.tokenizer.pad(
            clean_features,
            padding=True,
            return_tensors="pt",
        )
        clean_features['label']=torch.tensor(labels,dtype=torch.long)
        return clean_features
