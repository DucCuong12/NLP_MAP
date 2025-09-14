

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
    response=""

    #them vao skip_special_tokens=True
    start = inputs["input_ids"].shape[1]
    return tokenizer_cot.decode(generated_outputs[0, start:], skip_special_tokens=True)

def infer(
    row,
    batch_size,
    labels,
    tokenizer_re_ranker,
    model_re_ranker,
    thought,
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
    correct_df,
    decode_steps,
    num_of_top_retrieve_result,
    num_of_top_final_result,
    alpha,
    beta,
    tokenizer_retrieve,
    model_retrieve,
    infer_embeds,
    infer_labels,
    tokenizer_cot,
    model_cot,
    tokenizer_re_ranker,
    model_re_ranker,
    yes_token_id,
    yes_g_token_id,
    no_token_id,
    no_g_token_id,
    map_ks=None   # NEW: list of K values for MAP@K
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

        candidate_labels, candidate_labels_scores = retrieve(
            row, tokenizer_retrieve, model_retrieve, num_of_top_retrieve_result, infer_embeds, infer_labels
        )

        thought = generate_cot(row, tokenizer_cot, model_cot)

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
            tokenizer_re_ranker,
            model_re_ranker,
            thought,
            decode_steps,
            yes_token_id,
            yes_g_token_id,
            no_token_id,
            no_g_token_id
        )

        sorted_candidate_desc, _ = ensemble(candidate_labels_scores, ranking_scores, alpha, beta)
        ranked_labels_full = [candidate_labels[k] for k in sorted_candidate_desc[:max_needed_k]]
        predictions_per_row.append(ranked_labels_full)

        if idx < 3:
            print("Top (debug):", ranked_labels_full[:num_of_top_final_result])

    # MAP@K scores
    map_scores = cal_map_ks(predictions_per_row, correct_df["Category:Misconception"], map_ks)

    # print("MAP scores:", " ".join([f"K={k}:{map_scores[k]:.4f}" for k in map_ks]))

    return map_scores

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

if __name__ == "__main__":
    #hyperparameters
    # MAP@K , K = 1,2,3,4,5... ; num_of_retrieve ; alpha , beta for ensembling
    #num_top_final_result = k
    ks= [1,2,3,4,5,6,7,8,9,10]
    num_of_top_final_result=10
    alpha = 0.3
    beta = 0.7
    num_of_top_retrieve_result = 10

    #quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 
    )


    #finetuned model : cot , re-rank
    #cot model
    MODEL_COT =""
    #re-ranker model
    MODEL_PATH= ""

    #pretrained model : 2 con Qwen3 8B , hoac co the dung chung 1 model Qwen3 8B
    PRETRAINED_RE_RANKER_MODEL = ""
    PRETRAINED_COT_MODEL = ""

    decode_steps=0

    
    # Config lai ten model / model path / data path
    ##Load retrieval model
    
    state_dict = torch.load("best_model_proposed_110.pth", map_location="cuda")
    model_retrieve_path = 'mathbert_model/transformers/default/1'
    tokenizer_retrieve_path = 'mathbert_tokenizer/transformers/default/1'
    tokenizer_retrieve = AutoTokenizer.from_pretrained(tokenizer_retrieve_path)
    model_retrieve = AutoModel.from_pretrained(model_retrieve_path).to("cuda")
    model_retrieve = torch.nn.DataParallel(model_retrieve)
    model_retrieve.load_state_dict(state_dict)

    embedded_training_dataset_path = 'embeddings_test190.csv'
    infer_data = pd.read_csv(embedded_training_dataset_path)
    
    infer_embeds_list = infer_data['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
    infer_embeds = torch.stack(list(infer_embeds_list.values))
    infer_embeds = F.normalize(infer_embeds, dim=1).to(device)
    infer_labels = infer_data['label']


    ####### Load dataset for evaluation #########

    #chua biet 2 ham nay dung lam gi
    def tok(row):
        text = row['text_train']
        return tokenizer_retrieve(text, truncation=True, padding='max_length', max_length=512)
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }

    #test_data
    test= pd.read_csv("test.csv",keep_default_na=False)
    test_dataset = Dataset.from_pandas(test)

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

    # dataset = Dataset.from_pandas(df)
    # temp_dataset= dataset
    # dataset = dataset.map(tok, batch= True, remove_columns=[col for col in dataset.column_names if col != 'label'])
    # dataset.set_format(type="torch")
    # infer_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    # print(len(infer_dataloader))
    

    ######## INFER for our proposed model retrieval + cot + reranker #########


    # ######## INFER for Retrieval model #########
    # model_name = 'tbs17/MathBERT'
    # model_retrieval_path =""
    # model_retrieval = AutoModel.from_pretrained(model_name)
    # tokenizer_retrieval = AutoTokenizer.from_pretrained(model_name)

    # model_retrieval = torch.nn.DataParallel(model_retrieval, device_ids=list(range(torch.cuda.device_count())))

    # state_dict = torch.load('checkpoints/best_model_proposed_110.pth')
    # model_retrieval.load_state_dict(state_dict) 

    # result = infer(model_retrieval, infer_dataloader)
    # save_embeddings_to_csv(result, 'embeddings_retrieval_test_split.csv')
    
    ####### INFER for Re-ranker model #########

    #Load CoT Model
    tokenizer_cot = AutoTokenizer.from_pretrained(MODEL_COT)
    if tokenizer_cot.pad_token is None:
        tokenizer_cot.pad_token = tokenizer_cot.eos_token
    model_cot = AutoModelForCausalLM.from_pretrained(
    MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    #Load Re-ranker Model
    tokenizer_re_rank = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer_re_rank.pad_token is None:
        tokenizer_re_rank.pad_token = tokenizer_re_rank.eos_token
    
    model_re_rank = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    #yes,no id
    yes_token_id = tokenizer_re_rank.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer_re_rank.encode("No", add_special_tokens=False)[0]

    yes_g_token_id = tokenizer_re_rank.encode(" Yes",add_special_tokens=False)[0]
    no_g_token_id = tokenizer_re_rank.encode(" No", add_special_tokens=False)[0]

    #infer
    map_k_scores_proposed = predict(test_dataset,
                           correct_df,
                           decode_steps,
                           num_of_top_retrieve_result,
                           num_of_top_final_result,
                           alpha,
                           beta,
                           tokenizer_retrieve,
                           model_retrieve,
                           infer_embeds,
                           infer_labels,
                           tokenizer_cot,
                           model_cot,
                           tokenizer_re_rank,
                           model_re_rank,
                           yes_token_id,
                           yes_g_token_id,
                           no_token_id,
                           no_g_token_id,
                           ks)
    print(str("MAP@K scores for our proposed model:\n" + str(map_k_scores_proposed)))
    with open("map_scores.txt","a") as f:
        f.write("MAP@K scores for our proposed model"+str(map_k_scores_proposed) + "\n")
    
    

####### INFER for Re-ranker without fine-tune CoT  #########
    decode_steps = 0
    pre_tokenizer_cot = AutoTokenizer.from_pretrained(PRETRAINED_COT_MODEL)
    if pre_tokenizer_cot.pad_token is None:
        pre_tokenizer_cot.pad_token = tokenizer_cot.eos_token
    pre_model_cot = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_COT_MODEL, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )
    #infer with pretrained cot and fine-tuned re-ranker
    map_k_scores_without_finetuned_cot = predict(test_dataset,
                           correct_df,
                           decode_steps,
                           num_of_top_retrieve_result,
                           num_of_top_final_result,
                           alpha,
                           beta,
                           tokenizer_retrieve,
                           model_retrieve,
                           infer_embeds,
                           infer_labels,
                           pre_tokenizer_cot,
                           pre_model_cot,
                           tokenizer_re_rank,
                           model_re_rank,
                           yes_token_id,
                           yes_g_token_id,
                           no_token_id,
                           no_g_token_id,
                           ks)
    print(str("MAP@K scores w/o finetuned CoT model:\n" + str(map_k_scores_without_finetuned_cot)))
    with open("map_scores.txt","a") as f:
        f.write("MAP@K scores w/o finetuned CoT model:\n"+str(map_k_scores_without_finetuned_cot) + "\n")

####### INFER for Re-ranker without fine-tune Yes/No  #########
    decode_steps = 0
    pre_tokenizer_re_rank = AutoTokenizer.from_pretrained(PRETRAINED_RE_RANKER_MODEL)
    if pre_tokenizer_re_rank.pad_token is None:
        pre_tokenizer_re_rank.pad_token = tokenizer_re_rank.eos_token
    
    pre_model_re_rank = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_RE_RANKER_MODEL, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )
    #infer with fine-tuned cot and pretrained re-ranker
    map_k_scores_withoud_finetuned_re_ranker = predict(test_dataset,
                           correct_df,
                           decode_steps,
                           num_of_top_retrieve_result,
                           num_of_top_final_result,
                           alpha,
                           beta,
                           tokenizer_retrieve,
                           model_retrieve,
                           infer_embeds,
                           infer_labels,
                           tokenizer_cot,
                           model_cot,
                           pre_tokenizer_re_rank,
                           pre_model_re_rank,
                           yes_token_id,
                           yes_g_token_id,
                           no_token_id,
                           no_g_token_id,
                           ks)
    print(str("MAP@K scores w/o finetuned re-ranker model:\n" + str(map_k_scores_withoud_finetuned_re_ranker)))
    with open("map_scores.txt","a") as f:
        f.write("MAP@K scores w/o finetuned re-ranker model:\n"+str(map_k_scores_withoud_finetuned_re_ranker) + "\n")
