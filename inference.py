

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
    
# def retrieve_all(test_dataset, tokenizer, model, num_of_top_result, infer_embeds, infer_labels):
#     texts = test_dataset['text_train']
#     tokenized_texts = tokenizer(
#         texts,
#         truncation=True,
#         padding='max_length',
#         max_length=512,
#         return_tensors="pt"
#     ).to(device)
#     with torch.no_grad():
#         output = model(**tokenized_texts)
#     queries = output.last_hidden_state[:, 0, :]
#     query = F.normalize(queries, dim=1)
#     sims = torch.matmul(queries, infer_embeds.T).squeeze(0)
#     sorted_idx = torch.argsort(sims, descending=True)
#     seen_labels = set()
#     top_results = []
#     top_results_score = []
#     for idx in sorted_idx.tolist():
#         lbl = infer_labels[idx]
#         if lbl not in seen_labels:
#             score = sims[idx].item()
#             top_results.append(lbl)
#             top_results_score.append(score)
#             seen_labels.add(lbl)
#         if len(top_results) == num_of_top_result:
#             break
#     return top_results, top_results_score

def retrieve_all(
    test_dataset,
    tokenizer,
    model,
    num_of_top_result,
    infer_embeds,
    infer_labels,
    batch_size=32
):
    texts = test_dataset['text_train']  
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
    test_dataset.add_column("candidate_labels",all_candidate_labels)
    test_dataset.add_column("candidate_label_scores",all_candidate_scores)

    return test_dataset,all_candidate_labels, all_candidate_scores

def tokenize_input_for_cot(row,tokenizer) :
    
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
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
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

<|im_end|>
<|im_start|>assistant
"""
    message = [
       {"role" : "user" , "content" :prompt}
   ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_output = tokenizer(
        text,
        truncation = True,
        max_length = 2048,
        return_tensors = "pt"
    )
    
    return tokenized_output.to(device)

def tokenize_input(row, category, misconception, tokenizer,thought):
    
    
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
    parts = category.split("_")
    correctness = parts[0]
    reasoning_type = parts[1]
    prompt = f"""
<|im_start|>system
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem , then a THOUGHT ANALYSIS and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the your knowledge and problem data. Note that the THOUGHT ANALYSIS may be sometimes not correct.

DEFINITIONS OF THE CLASSIFICATION LABELS:

Part 1: Correctness (True or False): This describes whether the student's answer is objectively the correct solution to the Question Text.

Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
Correct: The explanation shows sound, logical, and mathematically valid reasoning.
Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

Part 3: Misconception : This is a text description of the specific thinking error. It is only relevant when the ReasoningType is Misconception. If the ReasoningType is Correct or Neither, this field's value should be "NA".

YOUR TASK:

1. Compare the THOUGHT ANALYSIS to the Correctness,ReasoningType and Misconception in PROPOSED CLASSIFICATION based on PROBLEM DATA, then consider the following 3 questions :
1.1 From the THOUGHT ANALYSIS and your analysis, does the true "True/False" conclusion of student's answer match the Correctness value in PROPOSED CLASSIFICATION?
1.2 Does the true "Correct/Misconception/Neither" conclusion match the ReasoningType value in PROPOSED CLASSIFICATION?
1.3 If the ReasoningType value in PROPOSED CLASSIFICATION  is "Misconception", does the student's error align with the provided Misconception" text? (If the "ReasoningType" value in PROPOSED CLASSIFICATION is Correct or Neither, you can skip this step) 
2. Final Conclusion: A "Yes" is only possible if all checks in Step 1 pass. If there is any mismatch at any point, the answer must be "No".

**CONSTRAINT:
You are only allowed to output only one token ("Yes"/"No").


<|im_end|>
<|im_start|>user
PROBLEM DATA:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

PROPOSED CLASSIFICATION:
Correctness: '{correctness}'
ReasoningType: '{reasoning_type}'
Misconception: '{misconception}'

THOUGHT ANALYSIS:
{thought}

<|im_end|>
<|im_start|>assistant
"""
    message = [
       {"role" : "user" , "content" :prompt}
   ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_output = tokenizer(
        text,
        truncation = True,
        max_length = 1024,
        return_tensors = "pt"
    )
    
    return tokenized_output.to(device)

def generate_cot(row,tokenizer_cot,model_cot) :
    inputs = None
    with torch.no_grad():
        inputs = tokenize_input_for_cot(row,tokenizer_cot)
        generated_outputs = model_cot.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer_cot.eos_token_id
        )

    #them vao skip_special_tokens=True
    start = inputs["input_ids"].shape[1]
    return tokenizer_cot.decode(generated_outputs[0, start:], skip_special_tokens=True)

def infer(
    row,
    batch_size,
    labels,
    thought,
    tokenizer_re_ranker,
    model_re_ranker,
    decode_steps,
    yes_token_id,
    yes_g_token_id,
    no_token_id,
    no_g_token_id,
    sub_batch_size=7  # Add this argument
):
    categories, misconceptions = labels
    assert len(categories) == batch_size and len(misconceptions) == batch_size


    all_scores = []
    n = batch_size
    for start in range(0, n, sub_batch_size):
        end = min(start + sub_batch_size, n)
        input_dicts = []
        for i in range(start, end):
            input_dict = tokenize_input(row, categories[i], misconceptions[i], tokenizer_re_ranker, thought)
            input_dicts.append(input_dict)
        
        batch = {}
        for key in input_dicts[0]:
            batch[key] = torch.cat([d[key] for d in input_dicts], dim=0).to(device)
            
        with torch.no_grad():
            outputs = model_re_ranker(**batch)

        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        yes_logits = torch.max(
            last_token_logits[:, yes_token_id],
            last_token_logits[:, yes_g_token_id]
        )
        no_logits = torch.max(
            last_token_logits[:, no_token_id],
            last_token_logits[:, no_g_token_id]
        )

        scores = yes_logits - no_logits
        all_scores.append(scores.cpu())
        if decode_steps < 3:
            print(f"--- Sub-batch {start}-{end} scores: {scores} ---")
        decode_steps += 1

    # Concatenate all sub-batch scores
    all_scores = torch.cat(all_scores, dim=0).numpy()
    return all_scores.tolist(), decode_steps

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensemble(retrieve_scores,re_rank_scores,alpha,beta):
    retrieve_scores = np.array(retrieve_scores)
    re_rank_scores = np.array(re_rank_scores)
    retrieve_softmax = softmax(retrieve_scores)
    re_rank_softmax = softmax(re_rank_scores)
    
    combined = alpha * retrieve_softmax + beta * re_rank_softmax
    
    indices_sorted = np.argsort(-combined) 
    combined_scores_sorted = combined[indices_sorted]
    return indices_sorted.tolist(), combined_scores_sorted.tolist()


def predict(
    dataset,
    correct_labels,
    decode_steps,
    num_of_top_final_result,
    alpha,
    beta,
    finetuned_cot,
    tokenizer_re_ranker,
    model_re_ranker,
    yes_token_id,
    yes_g_token_id,
    no_token_id,
    no_g_token_id,
    map_ks   # list of K values for MAP@K
):
    
    #Returns map_scores: dict {K: MAP@K}
    
    if map_ks is None:
        map_ks = [num_of_top_final_result]
    max_needed_k = max(map_ks + [num_of_top_final_result])

    predictions_per_row = []
    U = len(dataset)

    for idx, row in enumerate(dataset):
        if idx < 3:
            print(f"--This is the {idx} test---")

        candidate_labels, candidate_labels_scores = row["candidate_labels"] , row["candidate_label_scores"]

        if finetuned_cot == True :
            thought = row["finetuned_cot_response"]
        else :
            thought = row["pre_cot_response"]
            

        categories = []
        misconceptions = []
        for label in candidate_labels:
            parts = label.split(":")
            # safety
            if len(parts) < 2:
                categories.append(parts[0])
                misconceptions.append("NA")
            else:
                categories.append(parts[0])
                misconceptions.append(parts[1])
        labels = (categories, misconceptions)
        batch_size = len(candidate_labels)

        ranking_scores, decode_steps = infer(
            row,
            batch_size,
            labels,
            thought,
            tokenizer_re_ranker,
            model_re_ranker,
            decode_steps,
            yes_token_id,
            yes_g_token_id,
            no_token_id,
            no_g_token_id,
            sub_batch_size=7
        )

        sorted_candidate_desc, _ = ensemble(candidate_labels_scores, ranking_scores, alpha, beta)
        ranked_labels_full = [candidate_labels[k] for k in sorted_candidate_desc[:max_needed_k]]
        predictions_per_row.append(ranked_labels_full)

        if idx < 3:
            print("Top (debug):", ranked_labels_full[:num_of_top_final_result])

    # MAP@K scores
    map_scores = cal_map_ks(predictions_per_row, correct_labels, map_ks)

    # print("MAP scores:", " ".join([f"K={k}:{map_scores[k]:.4f}" for k in map_ks]))

    return map_scores,predictions_per_row

def tokenize_input_for_infer_only(row, category, misconception, tokenizer):
    
    
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
    parts = category.split("_")
    correctness = parts[0]
    reasoning_type = parts[1]
    prompt = f"""
<|im_start|>system
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem , then a THOUGHT ANALYSIS and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the your knowledge and problem data. Note that the THOUGHT ANALYSIS may be sometimes not correct.

DEFINITIONS OF THE CLASSIFICATION LABELS:

Part 1: Correctness (True or False): This describes whether the student's answer is objectively the correct solution to the Question Text.

Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
Correct: The explanation shows sound, logical, and mathematically valid reasoning.
Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

Part 3: Misconception : This is a text description of the specific thinking error. It is only relevant when the ReasoningType is Misconception. If the ReasoningType is Correct or Neither, this field's value should be "NA".

YOUR TASK:

1. Consider the following 3 questions :
1.1 From your analysis, does the true "True/False" conclusion of student's answer match the Correctness value in PROPOSED CLASSIFICATION?
1.2 Does the true "Correct/Misconception/Neither" conclusion match the ReasoningType value in PROPOSED CLASSIFICATION?
1.3 If the ReasoningType value in PROPOSED CLASSIFICATION  is "Misconception", does the student's error align with the provided Misconception" text? (If the "ReasoningType" value in PROPOSED CLASSIFICATION is Correct or Neither, you can skip this step) 
2. Final Conclusion: A "Yes" is only possible if all checks in Step 1 pass. If there is any mismatch at any point, the answer must be "No".

**CONSTRAINT:
You are only allowed to output only one token ("Yes"/"No").


<|im_end|>
<|im_start|>user
PROBLEM DATA:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

PROPOSED CLASSIFICATION:
Correctness: '{correctness}'
ReasoningType: '{reasoning_type}'
Misconception: '{misconception}'

<|im_end|>
<|im_start|>assistant
"""
    message = [
       {"role" : "user" , "content" :prompt}
   ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_output = tokenizer(
        text,
        truncation = True,
        max_length = 1024,
        return_tensors = "pt"
    )
    
    return tokenized_output.to(device)

def infer_reranker_only(dataset,
                        training_df_path,
                        tokenizer_re_ranker,
                        model_re_ranker,
                        yes_token_id,
                        yes_g_token_id,
                        no_token_id,
                        no_g_token_id,              # used for prompt tokenization inside tokenize_input_for_infer_only
                        sub_batch_size=7):


    training_df = pd.read_csv(training_df_path, keep_default_na=False)

    all_misconceptions = []
    for item in training_df['Misconception']:
        item_str = str(item).strip()
        if item_str != "NA" and item_str not in all_misconceptions:
            all_misconceptions.append(item_str)

    candidate_labels = ["True_Correct:NA", "True_Neither:NA", "False_Correct:NA", "False_Neither:NA"]
    for item in all_misconceptions:
        candidate_labels.append(f"True_Misconception:{item}")
        candidate_labels.append(f"False_Misconception:{item}")

    categories = []
    misconceptions = []
    for item in candidate_labels:
        cat, mis = item.split(":", 1)
        categories.append(cat)
        misconceptions.append(mis)

    n_candidates = len(candidate_labels)
    all_scores_per_row = []

    for idx, row in enumerate(dataset):
        per_sample_scores = []
        for start in range(0, n_candidates, sub_batch_size):
            end = min(start + sub_batch_size, n_candidates)
            input_dicts = []
            for i in range(start, end):
                cat_i = categories[i]
                mis_i = misconceptions[i]
                input_dict = tokenize_input_for_infer_only(row, cat_i, mis_i, tokenizer_re_ranker)
                input_dicts.append(input_dict)

            batch = {}
            for key in input_dicts[0]:
                batch[key] = torch.cat([d[key] for d in input_dicts], dim=0).to(device)

            with torch.no_grad():
                outputs = model_re_ranker(**batch)

            logits = outputs.logits
            last_token_logits = logits[:, -1, :]

            yes_logits = torch.max(
                last_token_logits[:, yes_token_id],
                last_token_logits[:, yes_g_token_id]
            )
            no_logits = torch.max(
                last_token_logits[:, no_token_id],
                last_token_logits[:, no_g_token_id]
            )

            scores = (yes_logits - no_logits).cpu().tolist()  # list of floats for this sub-batch
            per_sample_scores.extend(scores)

        if len(per_sample_scores) != n_candidates:
            per_sample_scores = (per_sample_scores + [0.0] * n_candidates)[:n_candidates]

        # order_desc = sorted(range(len(per_sample_scores)), key=lambda i: per_sample_scores[i], reverse=True)
        # per_sample_scores_desc = [per_sample_scores[i] for i in order_desc]
        order_desc = sorted(range(len(per_sample_scores)), key=lambda i: per_sample_scores[i], reverse=True)
        per_sample_scores_desc = [per_sample_scores[i] for i in order_desc]
        per_sample_labels_desc = [candidate_labels[i] for i in order_desc]

        all_scores_per_row.append(per_sample_labels_desc)

    return all_scores_per_row

        

        


#not used
def cal_map_k(k,test_df,correct_df):
    sum = 0
    count = 0
    for idx, row in test_df.iterrows() :
        count +=1
        candidate_labels = row["Category:Misconception"].split(" ")
        for i,label in enumerate(candidate_labels):
            if i >= k:
                break
            if label == correct_df.loc[idx,"Category:Misconception"]:
                sum += float(1/(i+1))
                break

    sum = float(sum/count)
    return sum

def cal_map_ks(predictions_per_row, correct_labels, ks):
   
    # Normalize ks
    if not ks:
        return {}

    if hasattr(correct_labels, "reset_index"):
        correct_labels = correct_labels.reset_index(drop=True)
    if hasattr(correct_labels, "tolist"):
        correct_labels = correct_labels.tolist()
    n = len(predictions_per_row)
    if n == 0:
        return {k: 0.0 for k in ks}

    max_k = ks[-1]
    sums = {k: 0.0 for k in ks}

    for i in range(n):
        preds = predictions_per_row[i]
        true_label = correct_labels[i]
        truncated = preds[:max_k]
        match_rank = None
        for r, lbl in enumerate(truncated):
            if lbl == true_label:
                match_rank = r
                break
        if match_rank is not None:
            for k in ks:
                if match_rank < k:
                    sums[k] += 1.0 / (match_rank + 1)

    return {k: (sums[k] / n) for k in ks}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["retrieval_only", "rerank_only", "proposed_model","proposed_model_wo_finetuned_cot","proposed_model_wo_finetuned_rerank"], default="proposed_model")
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--epochs", type=int, default=10)

    return parser.parse_args()




if __name__ == "__main__":
    #hyperparameters
    # MAP@K , K = 1,2,3,4,5... ; num_of_retrieve ; alpha , beta for ensembling
    #num_top_final_result = k
    #training_csv_path : training_path before spliting
    ks= [1,3,5]
    num_of_top_final_result=10
    alpha = 0.3
    beta = 0.7
    num_of_top_retrieve_result = 10
    training_df_path = ""

    #quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 
    )
    #finetuned reranker model (with finetuned cot model) :
    MODEL_PATH= ""
    
    #finetuned reranker model (without pre-trained cot model)
    MODEL_WO_FT_COT=""

    #finetuned reranker model (without cot model)
    MODEL_WO_COT=""

    #pretrained reranker model (with finetuned cot model)
    MODEL_PRETRAIN_W_FT_COT = ""

    #pretrained model : 2 con Qwen3 8B , hoac co the dung chung 1 model Qwen3 8B
    

    PRETRAINED_RE_RANKER_MODEL = ""
    decode_steps=0
    


    #test_data
    test_df= pd.read_csv('test.csv',keep_default_na=False)
    test_dataset = Dataset.from_pandas(test_df)

    #ground_truth 
    data_path = 'test_split.csv'
    df = pd.read_csv(data_path,keep_default_na=False)
    correct_df = df[['Category', 'Misconception']].copy()
    correct_df['Category'] = correct_df['Category'].astype(str).str.strip()
    correct_df['Misconception'] = (
        correct_df['Misconception']
        .replace(['', ' ', None], 'NA')
        .fillna('NA')
        .astype(str)
        .str.strip()
    )
    correct_df['Category:Misconception'] = (
        correct_df['Category'] + ':' + correct_df['Misconception']
    )
    correct_df = correct_df.reset_index(drop=True)
    assert len(correct_df) == len(test_dataset), f"len(correct_df)={len(correct_df)} != len(test_dataset)={len(test_dataset)}"
    correct_labels = correct_df['Category:Misconception'].reset_index(drop=True)

    
    ############ arg --mode

    args = get_args()


    ######## INFER for retrieval only #########
    if args.mode == "retrieval_only":
        infer_result_labels = test_df["candidate_labels"]

    # ######## INFER for rerank only ############
    # if args.mode == "rerank_only":

    #     #load model rerank(finetuned)
    #     tokenizer = AutoTokenizer.from_pretrained(MODEL_WO_COT)
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token
        
    #     model = AutoModelForCausalLM.from_pretrained(
    #         MODEL_WO_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    #     )
    #     yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    #     no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    #     yes_g_token_id = tokenizer.encode(" Yes",add_special_tokens=False)[0]
    #     no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    #     #infer
    #     infer_result_labels = infer_reranker_only(
    #         test_dataset,
    #         training_df_path,
    #         tokenizer,
    #         model,
    #         yes_token_id,
    #         yes_g_token_id,
    #         no_token_id,
    #         no_g_token_id,
    #         sub_batch_size=5,
    #     )

    ######## INFER for proposed model ###################
    if args.mode == "proposed_model":


        #load rerank model(finetuned)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model= AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
        )
        finetuned_cot = True

    ####### INFER for Re-ranker without fine-tune CoT  #########

    if args.mode == "proposed_model_wo_finetuned_cot":

        #load re-rank model
        tokenizer= AutoTokenizer.from_pretrained(MODEL_WO_FT_COT)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model= AutoModelForCausalLM.from_pretrained(
            MODEL_WO_FT_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
        )
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
        yes_g_token_id = tokenizer.encode(" Yes",add_special_tokens=False)[0]
        no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
        finetuned_cot = False

    ####### INFER for Re-ranker without fine-tune ReRanker  #########
    if args.mode == "proposed_model_wo_finetuned_rerank":

        
        #load pre_finetuned_rerank_model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAIN_W_FT_COT)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model= AutoModelForCausalLM.from_pretrained(
            MODEL_PRETRAIN_W_FT_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
        )

        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
        yes_g_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
        finetuned_cot = True
        
        
    if args.mode not in ["retrieval_only","rerrank_only"] :
        map_scores = predict(
            test_dataset,
            correct_labels,
            decode_steps,
            num_of_top_final_result,
            alpha,
            beta,
            finetuned_cot,
            tokenizer,
            model,
            yes_token_id,
            yes_g_token_id,
            no_token_id,
            no_g_token_id,
            ks
        )
    
    # khong dung rerank only


    if args.mode in ["retrieval_only","rerank_only"]:
        map_scores = cal_map_ks(infer_result_labels,correct_labels,ks)


    print("MAP@K scores for "+str(args.mode)+" is: " + str(map_scores))
    with open("map_scores.txt","a") as f:
        f.write("MAP@K scores for "+str(args.mode)+" is: " + str(map_scores))






