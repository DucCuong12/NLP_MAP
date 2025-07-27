import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from torch.amp import autocast, GradScaler
from process_data import prepare_df, prepare_data
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# === Stage A: Embedding Model Pretraining ===

os.chdir('/bigdisk/cuongvd17/Testing/kaggle')
df = prepare_df('train.csv')

dataset = Dataset.from_pandas(df)
# Prepare text and label fields
dataset = dataset.map(prepare_data, remove_columns=dataset.column_names)

# ----------------------------------------
# 2. Continual MLM pretraining
# ----------------------------------------
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_mlm(example):
    return tokenizer(example['text'],
                     truncation=True,
                     padding='max_length',
                     max_length=256)

mlm_dataset = dataset.map(tokenize_mlm, batched=True, remove_columns=dataset.column_names)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Instantiate model for MLM
mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

# Training args for MLM
mlm_args = TrainingArguments(
    output_dir='./stageA_mlm',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=16, 
    dataloader_pin_memory=True 
    
)

trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=mlm_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting MLM pretraining...")
trainer.train()

print("MLM pretraining finished.")
# Save adapted encoder
mlm_model.save_pretrained('./stageA_mlm')

# ----------------------------------------
# 3. Contrastive embedding training
# ----------------------------------------
from transformers import AutoModel
encoder = AutoModel.from_pretrained('./stageA_mlm')
encoder.eval()

def contrastive_training(encoder, dataset, epochs=3, batch_size=64, lr=1e-5, temperature=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    scaler = GradScaler('cuda')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True)
    for epoch in range(epochs):
        print(epochs)
        for batch in loader:
            texts = batch['text']
            labels = batch['label'].to(device)
            # tokenize
            enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            for k in enc:
                enc[k] = enc[k].to(device)
            # forward pass
            with autocast('cuda'):
           
                hidden = encoder(**enc)  # (B, D)
                hidden = hidden.last_hidden_state[:, 0, :]  
                # Normalize embeddings
                emb = F.normalize(hidden, dim=-1)
                # similarity matrix
                sim = (emb @ emb.T / temperature)
                mask = torch.eye(sim.size(0), dtype=torch.bool).to(device)
                sim.masked_fill_(mask, torch.finfo(sim.dtype).min)
                # positive mask
                lbl = labels.unsqueeze(1) == labels.unsqueeze(0)
                
                lbl = lbl.to(sim.device) & ~mask  
                sim_exp = torch.exp(sim)
                numerator = (sim_exp * lbl).sum(dim=1)

                denominator = sim_exp.sum(dim=1)

                # Add eps for numerical stability
                eps = torch.finfo(sim.dtype).min
                loss = -torch.log((numerator + eps) / (denominator + eps)).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Save encoder
    encoder.module.save_pretrained('./stageA_contrastive')

# Run contrastive training
print('Starting contrastive training...')
contrastive_training(encoder, dataset)
