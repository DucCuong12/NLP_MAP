from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
import ast
import torch.nn.functional as F
import pandas as pd


# embedded__training_dataset_path = "embeddings_test190.csv"
# model_retrieve_path = "mathbert_model"
# tokenizer_retrieve_path = "mathbert_tokenizer"
# state_dict_path = "best_model_proposed_110.pth"


class RetrieveEmbedding :
    def __init__(self,embedded_training_dataset_path,model_retrieve_path,tokenizer_retrieve_path,state_dict_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dict = torch.load("best_model_proposed_110.pth", map_location="cuda")
        self.infer_data = pd.read_csv(embedded_training_dataset_path)
        self.model_retrieve = AutoModel.from_pretrained(model_retrieve_path).to("cuda")
        self.tokenizer_retrieve = AutoTokenizer.from_pretrained(tokenizer_retrieve_path)
        self.model_retrieve = torch.nn.DataParallel(self.model_retrieve).load_state_dict(self.state_dict)
        infer_embeds_list = self.infer_data['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
        self.infer_embeds = torch.stack(list(infer_embeds_list.values))
        self.infer_embeds = F.normalize(self.infer_embeds, dim=1).to(self.device)
        self.infer_labels = self.infer_data['label']

    def retrieve(self,row):
        device=self.device
        tokenizer = self.tokenizer_retrieve
        model = self.model_retrieve

        text = f"[CLS] Question: {row['QuestionText']}\n[SEP] Student's Answer: {row['MC_Answer']}\n[SEP] Student's explanation: {row['StudentExplanation']}[SEP]\n "
        tokenized_text= tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_tensors="pt"
                    )
        tokenized_text = tokenized_text.to(device)
        with torch.no_grad():
            output = model(**tokenized_text)
        query = output.last_hidden_state[:, 0, :]          
        query = F.normalize(query, dim=1)                 
            
                
        sims = torch.matmul(query, self.infer_embeds.T).squeeze(0)  
        
        sorted_idx = torch.argsort(sims, descending=True)
        
        seen_labels = set()
        top_results = []
        for idx in sorted_idx.tolist():
            lbl = self.infer_labels[idx]
            if lbl not in seen_labels:
                score = sims[idx].item()
                top_results.append(lbl)
                seen_labels.add(lbl)
            if len(top_results) == 10:
                break
        return top_results
        