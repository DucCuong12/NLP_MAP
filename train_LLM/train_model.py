from copy import deepcopy
from torch.optim import AdamW
import torch
import torch.distributed as dist
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
import datetime

from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
import numpy as np

# datasets.Dataset bị trùng tên với module dataset của bạn, nên tránh import *
# from datasets import Dataset  # không cần thiết ở đây
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import Config
from dataset import MathDataset
from collator import TextCollator  # nếu bạn muốn dùng custom collator
from evaluate import run_evaluation

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- Helpers ----------
def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup_ddp():
    """Initialize process group & return local_rank, world_size, device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Fallback single process (useful for debugging without torchrun)
        rank, world_size, local_rank = 0, 1, 0

    if dist.is_available() and not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return local_rank, world_size, device

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ---------- Main ----------
def run_main():
    local_rank, world_size, device = setup_ddp()

    if is_main_process():
        print(f"script_dir: {script_dir}")
        print(f"cwd: {os.getcwd()}")
        print(f"world_size (num GPUs): {world_size}")
        print(f"Using device: {device}")

    cfg = Config()
    # override cfg.device để đảm bảo mỗi process dùng đúng GPU cục bộ
    cfg.device = device

    if is_main_process():
        print("*"*50)
        print("Preparing data...")

    # Đọc & merge dữ liệu (mỗi process đều làm, nhưng chỉ rank 0 log)
    train_df = pd.read_csv("train.csv")
    explanation_df = pd.read_parquet("generated_result/teacher_outputs.parquet")
    explanation_df = explanation_df.drop(columns=['query_id'])
    train_df = train_df.rename(columns={"QuestionId": "query_id"})
    # Merge theo index như code gốc
    train_df = train_df.merge(explanation_df, left_index=True, right_index=True)

    # Split
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

    if is_main_process():
        print("Change to datasets...")

    dataset_creator = MathDataset(model_name=cfg.model_name)
    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)

    # Samplers cho DDP
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank() if dist.is_initialized() else 0, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=dist.get_rank() if dist.is_initialized() else 0, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Khởi tạo model 1 lần, Không dùng DataParallel
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    if cfg.use_lora:
        if is_main_process():
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

    model.to(cfg.device)

    # Bọc DDP nếu chạy nhiều GPU
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # thường để False cho CausalLM
        )

    # Data collator: truyền model.module nếu DDP để collator biết shift labels khi cần
    base_model_for_collator = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=base_model_for_collator,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    # Nếu bạn có collator riêng, có thể thay:
    # data_collator = TextCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    # Dataloaders (không dùng shuffle khi có sampler)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=8,
        drop_last=True,  # giúp các replica cân batch
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        sampler=valid_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
    )

    if is_main_process():
        print("*"*50)
        print("Preparing optimizer/scheduler...")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Tổng bước: theo sampler -> len(train_dl) đã chuẩn cho mỗi replica
    num_update_steps_per_epoch = len(train_dl) // max(1, cfg.gradient_accumulation_steps)
    num_training_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.warmup_pct * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    start_time = time.time()
    current_iteration = 0

    # Progress bar chỉ ở rank 0
    for epoch in range(cfg.epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n----------- Epoch {epoch + 1}/{cfg.epochs} -----------")

        model.train()
        total_loss = 0.0

        iterable = train_dl
        if is_main_process():
            iterable = tqdm(train_dl, desc=f"Epoch {epoch + 1} Training")

        for step, batch in enumerate(iterable):
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            # trung bình loss giữa các GPU để log đẹp (không bắt buộc cho backward)
            loss_for_log = loss.detach()

            # Gradient Accumulation
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                current_iteration += 1

            # Giữ total_loss để hiển thị
            total_loss += loss_for_log.item()

            # Chỉ rank 0 hiển thị
            if is_main_process():
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = total_loss / (step + 1)
                    if isinstance(iterable, tqdm):
                        iterable.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

            # Eval theo tần suất, chỉ rank 0 chạy evaluation để tránh trùng lặp
            if is_main_process() and cfg.eval_frequency > 0 and current_iteration > 0 and current_iteration % cfg.eval_frequency == 0:
                print(f"\nrun_evaluation: {current_iteration}...")
                # Nếu run_evaluation cần model gốc, truyền model.module khi bọc DDP
                eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                eval_response = run_evaluation(eval_model, valid_dl, cfg.device)
                valid_loss = eval_response.get("valid_loss", None)
                if valid_loss is not None:
                    print(f">>> valid_loss = {valid_loss:.4f}")
                model.train()

        # Lưu checkpoint mỗi epoch, chỉ rank 0
        if is_main_process():
            checkpoint_dir = os.path.join(script_dir, f"checkpoint-{epoch}")
            base_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            base_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[Rank0] Saved checkpoint to {checkpoint_dir}")

        # Đồng bộ các tiến trình trước khi sang epoch mới
        if dist.is_initialized():
            dist.barrier()

    # Save cuối cùng, chỉ rank 0
    if is_main_process():
        print("*"*50)
        print("save model")
        final_model_dir = os.path.join(script_dir, "final_model")
        base_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        base_to_save.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        total_time = time.time() - start_time
        print(f"Total time to train: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"save model: {final_model_dir}")

    cleanup_ddp()

if __name__ == "__main__":
    run_main()
