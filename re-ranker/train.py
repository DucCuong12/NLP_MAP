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


def train_ranker():
    MODEL_NAME= "Qwen/Qwen2.5-7B-Instruct"
    OUTPUT_DIR = "./qwen2-7b-ranker-finetuned"

    POSITVE_DF_PATH =""
    NEGATIVE_DF_PATH=""


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False # use cache

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token ID for 'Yes': {yes_token_id}, 'No': {no_token_id}")


    tokenize_data =TokenizeData(POSITVE_DF_PATH,NEGATIVE_DF_PATH)
    positive_dataset = tokenize_data.get_dataset(tokenize_data.positive_df,tokenizer)
    negative_dataset = tokenize_data.get_dataset(tokenize_data.negative_df,tokenizer)


    # conf training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=7, #batch size=2
        gradient_accumulation_steps=3, 
        # total = 7 x 3 = 21 samples in a batch

        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=0,
        report_to="tensorboard",
    )
    


    # initialize datacollator and trainer
    data_collator = RankerDataCollator(tokenizer=tokenizer,negative_dataset=negative_dataset)
    
    trainer = CrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=positive_dataset,
        data_collator=data_collator,
        
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        decode_steps = 0,
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