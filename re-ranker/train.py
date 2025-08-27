import transformers
import peft
import trl
from IPython.display import FileLink


import torch
import torch.nn as nn


import numpy as np
import pandas as pd

from datasets import Dataset , load_from_disk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from model import CrossEntropyTrainer
from collator import RankerDataCollator
from tokenization import TokenizeData
from cot import CotModel
from process_data import DataProcess
# from retrieve import RetrieveEmbedding

def train_ranker():

    # Qwen/Qwen3-8B
    MODEL_NAME= ""
    OUTPUT_DIR = "./qwen3-8b-ranker-finetuned"


    # model cot
    MODEL_COT = ""

    # EMBEDDED_DATASET_PATH = "embeddings_test190.csv"
    # RETRIEVAL_MODEL_PATH = "mathbert_model"
    # RETRIEVAL_TOKENIZER_PATH ="mathbert_tokenize"
    # STATE_DICT_PATH = "best_model_proposed_110.pth"

    TRAINING_DATASET="train_fixed.csv"
    NEW_TRAINING_DATASET ="training_dataset.csv"

    # model_retrieve = RetrieveEmbedding(EMBEDDED_DATASET_PATH,RETRIEVAL_MODEL_PATH,RETRIEVAL_TOKENIZER_PATH,STATE_DICT_PATH)
    
    data_process = DataProcess(TRAINING_DATASET,NEW_TRAINING_DATASET)
    df,labels = data_process.format_train_data()
    df = data_process.create_negative_train_data(df,labels)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    model_cot= CotModel(MODEL_COT,quantization_config=bnb_config)


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False 

    yes_g_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token ID for ' Yes': {yes_g_token_id}, ' No': {no_g_token_id}")
    print(f"Token ID for 'Yes': {yes_token_id}, 'No': {no_token_id}")

    
    queries = model_cot.tokenize_input_for_cot(df)
    thoughts = model_cot.generate_cot(queries,2)

    tokenize_data =TokenizeData(df)
    training_set = tokenize_data.get_dataset(thoughts,tokenizer)



    # conf training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=5, #batch size=2
        gradient_accumulation_steps=4, 

        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=0,
        report_to="tensorboard",
    )
    


    # initialize datacollator and trainer
    data_collator = RankerDataCollator(tokenizer=tokenizer)
    
    trainer = CrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        data_collator=data_collator,

        yes_g_token_id = yes_g_token_id,
        no_g_token_id = no_g_token_id,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        tokenizer =tokenizer,
    )

    # start_training
    print("Start fine-tuning với Cross-entropy Loss...\n")
    trainer.train()
    print("Training completed.\n")

    
    # save model
    final_adapter_dir = f"{OUTPUT_DIR}/final_adapters"
    trainer.save_model(final_adapter_dir)
    print(f"Adapters saved at: {final_adapter_dir}")



if __name__ == "__main__" :
    train_ranker()