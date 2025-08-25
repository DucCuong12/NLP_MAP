from transformers import DataCollatorWithPadding, get_cosine_schedule_with_warmup
import pandas as pd
from copy import deepcopy
class TextCollator(DataCollatorWithPadding):

    def __call__(self, features):
        labels = None
        if "label" in features[0].keys():
            labels = [feature['label'] for feature in features]

        features = [
            {
                "input_ids" : feature["input_ids"],
                "attention_mask" : feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding = "longest",
            max_length = self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        ft_len = batch["input_ids"].size(1)

        if labels is not None:
            pad_labels = []
            for label in labels:
                pad_label = [-100] * (ft_len - len(label)) + label
                pad_labels.append(pad_label)
            batch["labels"] =  torch.tensor(pad_labels, dtype=torch.int64)

        return batch