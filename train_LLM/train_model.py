from copy import deepcopy
from torch.optim import AdamW
import torch
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, get_cosine_schedule_with_warmup, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset


from config import Config
from dataset import MathDataset
from collator import TextCollator
from evaluate import run_evaluation

def run_main():
    cfg = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("*"*50)
    print("Preparing data...")
    train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    explanation_df = pd.read_parquet("/kaggle/input/teacher-llm/teacher_outputs.parquet")
    train_df = train_df.rename(columns={"QuestionId" : "query_id"})
    # Merge
    train_df = train_df.merge(
        explanation_df, on='query_id', how='inner'
    )
    train_df = train_df[:5000]
    # train_df sau khi merge sẽ có được cột explanation là CoT của con LLM to
    print("Split data...")
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print(train_df.columns)
    # Chuyển thành dataset
    print("Change to ds...")
    dataset_creator = MathDataset(model_name = cfg.model_name)
    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model, # Cần truyền model vào để nó biết cách shift labels nếu cần
    label_pad_token_id=-100, # Quan trọng: nói cho nó biết pad labels bằng -100
    pad_to_multiple_of=8
    )
    # collator_fn = TextCollator(tokenizer=tokenizer, pad_to_multiple_of=16)
    # Chuyển thành dataloader
    print("Change to dl...")
    train_dl = DataLoader(
        train_ds,
        batch_size = cfg.batch_size,
        shuffle = True,
        collate_fn = data_collator
    ) 
    valid_dl = DataLoader(
        valid_ds,
        batch_size = cfg.batch_size,
        shuffle = True,
        collate_fn = data_collator
    )

    print("*"*50)
    # Chuẩn bị model
    print("Preparing model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
    )

    if cfg.use_lora:
        print("Using LoRA...")
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=cfg.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.to(cfg.device)

    print("complete configuration...")
    print("*"*50)
        
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    num_update_steps_per_epoch = len(train_dl) // cfg.gradient_accumulation_steps
    num_training_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.warmup_pct * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    start_time = time.time()
    current_iteration = 0
    
    for epoch in range(cfg.epochs):
        print(f"\n----------- Epoch {epoch + 1}/{cfg.epochs} -----------")
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch + 1} Training")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss

            loss = loss / cfg.gradient_accumulation_steps
            
            loss.backward()
            
            total_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_iteration += 1
                
                lr = scheduler.get_last_lr()[0]
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if current_iteration > 0 and current_iteration % cfg.eval_frequency == 0:
                    print(f"\nrun_evaluation: {current_iteration}...")
                    eval_response = run_evaluation(model, valid_dl, cfg.device)
                    valid_loss = eval_response["valid_loss"]
                    print(f">>> valid_loss = {valid_loss:.4f}")
                    
                    checkpoint_dir = os.path.join("/kaggle/working", f"checkpoint-{current_iteration}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    model.train()

    # Save
    print("*"*50)
    print("save model")
    final_model_dir = os.path.join("/kaggle/working", "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    total_time = time.time() - start_time
    print(f"Total time to train: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"save model: {final_model_dir}")

if __name__ == "__main__":
    run_main()