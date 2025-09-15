

import numpy as np 
import pandas as pd
import os

import transformers
import peft
import trl

from sklearn.preprocessing import LabelEncoder
import ast

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel
from datasets import Dataset
import ast
import torch.nn.functional as F

from datasets import load_from_disk
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

from torch.utils.data import DataLoader
import argparse
from process_data import prepare_df
from fine_tune1 import infer,save_embeddings_to_csv
from copy import deepcopy

def prepare_df1(path):
    # path = args.path
    df = pd.read_csv(path,keep_default_na=False)
    df['text_train'] = df.apply(
    lambda row: f"[CLS] Question: {row['QuestionText']}\n[SEP] Student's Answer: {row['MC_Answer']}\n[SEP] Student's explanation: {row['StudentExplanation']}[SEP]\n ",
    axis=1
    )
    return df

def retrieve(test_row, tokenizer, model, num_of_top_result, infer_embeds, infer_labels):
    text = test_row['text_train']
    tokenized_text = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = model(**tokenized_text)
    query = output.last_hidden_state[:, 0, :]
    query = F.normalize(query, dim=1)
    sims = torch.matmul(query, infer_embeds.T).squeeze(0)
    sorted_idx = torch.argsort(sims, descending=True)
    seen_labels = set()
    top_results = []
    top_results_score = []
    for idx in sorted_idx.tolist():
        lbl = infer_labels[idx]
        if lbl not in seen_labels:
            score = sims[idx].item()
            top_results.append(lbl)
            top_results_score.append(score)
            seen_labels.add(lbl)
        if len(top_results) == num_of_top_result:
            break
    return top_results, top_results_score


def retrieve_all(
    test_df_path,
    tokenizer,
    model,
    num_of_top_result,
    infer_embeds,
    infer_labels,
    batch_size=32
):  
    returned_df = pd.read_csv(test_df_path,keep_default_na=False)
    df =  prepare_df1(test_df_path)
    texts  = df['text_train']
    total = len(texts)

    all_candidate_labels = []
    all_candidate_scores = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        tokenized = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**tokenized)
        
        # [B, H]
        queries = output.last_hidden_state[:, 0, :]
        queries = F.normalize(queries, dim=1)                  

     
        sims_batch = torch.matmul(queries, infer_embeds.T)    

        for row_idx in range(sims_batch.size(0)):
            sims = sims_batch[row_idx]
            sorted_idx = torch.argsort(sims, descending=True)

            seen_labels = set()
            top_results = []
            top_scores = []
            for cand_idx in sorted_idx.tolist():
                lbl = infer_labels[cand_idx]
                if lbl not in seen_labels:
                    top_results.append(lbl)
                    top_scores.append(sims[cand_idx].item())
                    seen_labels.add(lbl)
                if len(top_results) == num_of_top_result:
                    break

            all_candidate_labels.append(top_results)
            all_candidate_scores.append(top_scores)
    returned_df["candidate_labels"]= all_candidate_labels
    returned_df["candidate_label_scores"]=all_candidate_scores

    return returned_df

def tokenize_input_for_cot(test_df,tokenizer) :
    
    q_texts, mc_answers, explanations = test_df["QuestionText"], test_df["MC_Answer"], test_df["StudentExplanation"]
    prompts = []
    for i in range(len(test_df["QuestionText"])):
        prompt = f"""
        <|im_start|>system
        Response in maximum 512 words.
        You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem, and analyze it with respect to the question.
        Show your detailed reasoning by following these steps:
        YOUR STEP-BY-STEP VERIFICATION PROCESS (Chain-of-Thought):

        1. Analyze Answer Correctness (True/False Check): First, independently solve the math problem in the Question. Compare your result to the student's Answer. Is the student's answer objectively True (correct) or False (incorrect)?
        2. Analyze Explanation Quality (Reasoning Check): Now, ignore the final answer and focus only on the explanation.
        Deconstruct the student's logic. What steps did they follow? Based only on their text, classify their reasoning: Is it Correct, a clear Misconception, or Neither?
        If you identify a misconception, briefly describe it in your own words.

        Show your detailed reasoning by following these above 2 steps.

        <|im_end|>
        <|im_start|>user
        Problem Data:
        Question: {q_texts[i]}
        Student's Answer: {mc_answers[i]}
        Student's Explanation: {explanations[i]}

        <|im_end|>
        <|im_start|>assistant
        """
        prompts.append(prompt)

    messages=[]
    for prompt in prompts:
        message = [
        {"role" : "user" , "content" :prompt}
        ]
        messages.append(message)

    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_outputs = tokenizer(
        texts,
        truncation = True,
        max_length = 512,
        return_tensors = "pt"
    )
    
    return tokenized_outputs.to(device)


def generate_cot(inputs,tokenizer_cot,model_cot,sub_batch_size) :
    results = []
    total_items = inputs["input_ids"].size(0)

    for idx in range(0, total_items, sub_batch_size):
        end_idx = min(idx + sub_batch_size, total_items)
        mini_batch = {k: v[idx:end_idx].to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model_cot.generate(
                **mini_batch,
                max_new_tokens=512,
                pad_token_id=tokenizer_cot.eos_token_id,
                do_sample=False
            )

        prompt_len = mini_batch["input_ids"].shape[1]
        suffix_tokens = generated[:, prompt_len:]
        decoded_texts = tokenizer_cot.batch_decode(suffix_tokens, skip_special_tokens=True)
        results.extend(text.strip() for text in decoded_texts)
    return results


    # for start in range(0, n, sub_batch_size):
    #     end = min(start + sub_batch_size, n)
    #     input_dicts = []
    #     for i in range(start, end):
    #         input_dict = tokenize_input(row, categories[i], misconceptions[i], tokenizer_re_ranker, thought)
    #         input_dicts.append(input_dict)
        
    #     batch = {}
    #     for key in input_dicts[0]:
    #         batch[key] = torch.cat([d[key] for d in input_dicts], dim=0).to(device)
            
    #     with torch.no_grad():
    #         outputs = model_re_ranker(**batch)
    
    # for 
    #     with torch.no_grad():
    #         generated_outputs = model_cot.generate(
    #             **batch,
    #             max_new_tokens=512,
    #             pad_token_id=tokenizer_cot.eos_token_id
    #         )

    # #them vao skip_special_tokens=True
    # starts = inputs["input_ids"].shape[1]
    # decoded_ouptuts =  tokenizer_cot.decode(generated_outputs[0, starts:], skip_special_tokens=True)

def generate_retrieval_in_file(test_df_path,tokenizer,model):
    pass    
def generate_cot_in_file(test_df_path,tokenizer,model):
    pass
if __name__ == "__main__":

    test_df_path = ""
    PRE_MODEL_COT=""
    FINETUNED_MODEL_COT=""
    num_of_top_result=10

    test_df = pd.read_csv(test_df_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    #load finetuned cot model
    tokenizer_cot = AutoTokenizer.from_pretrained(FINETUNED_MODEL_COT)
    if tokenizer_cot.pad_token is None:
        tokenizer_cot.pad_token = tokenizer_cot.eos_token
    model_cot = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    #load pre cot model
    pre_tokenizer_cot = AutoTokenizer.from_pretrained(PRE_MODEL_COT)
    if pre_tokenizer_cot.pad_token is None:
        pre_tokenizer_cot.pad_token = pre_tokenizer_cot.eos_token
    pre_model_cot = AutoModelForCausalLM.from_pretrained(
    PRE_MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    #load retrieve model

    state_dict = torch.load("best_model_proposed_110.pth", map_location="cuda")
    model_retrieve_path = 'mathbert_model/transformers/default/1'
    tokenizer_retrieve_path = 'mathbert_tokenizer/transformers/default/1'
    tokenizer_retrieve = AutoTokenizer.from_pretrained(tokenizer_retrieve_path)
    model_retrieve = AutoModel.from_pretrained(model_retrieve_path).to("cuda")
    model_retrieve = torch.nn.DataParallel(model_retrieve)
    model_retrieve.load_state_dict(state_dict)

    embedded_training_dataset_path = ""
    infer_data = pd.read_csv(embedded_training_dataset_path)
    
    infer_embeds_list = infer_data['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
    infer_embeds = torch.stack(list(infer_embeds_list.values))
    infer_embeds = F.normalize(infer_embeds, dim=1).to(device)
    infer_labels = infer_data['label']

    #get retrieve candidates to test_df

    test_df = retrieve_all (
                            test_df_path,
                            tokenizer_retrieve,
                            model_retrieve,
                            num_of_top_result,
                            infer_embeds,
                            infer_labels,
                            batch_size=32
                            )  


    #get finetuned_cot responses to test_df
    tokenized_outputs = tokenize_input_for_cot(test_df,tokenizer_cot)
    results = generate_cot_in_file(test_df,tokenizer_cot,model_cot)
    test_df["finetuned_cot_response"] = results
    test_df["finetuned_cot_response"] = test_df['finetuned_cot_response'].astype('str')
    
    #get pre_cot responses to test_df
    pre_tokenized_outputs = tokenize_input_for_cot(test_df,pre_tokenizer_cot)
    pre_results = generate_cot_in_file(test_df,pre_model_cot)
    test_df["pre_cot_response"] = pre_results
    test_df["pre_cot_response"]= test_df["pre_cot_response"].astype('str')

    #save to csv
    test_df.to_csv(test_df_path)



    # generate_cot_in_file(test_df_path,)
    # generate_retrieval_in_file(test_df_path)
    # generate_cot_in_file(test_df_path,)