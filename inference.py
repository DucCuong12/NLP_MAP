

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
from transformers import AutoModel, AutoTokenizer
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
# import numpy as np

k=3

def prepare_df1(path):
    # path = args.path
    df = pd.read_csv(path)
    df['text_train'] = df.apply(
    lambda row: f"[CLS] Question: {row['QuestionText']}\n[SEP] Student's Answer: {row['MC_Answer']}\n[SEP] Student's explanation: {row['StudentExplanation']}[SEP]\n ",
    axis=1
    )
    return df

def retrieve(test_row,tokenizer,model):
    text = test_row['text_train']
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
        
            
    sims = torch.matmul(query, infer_embeds.T).squeeze(0)  
    
    sorted_idx = torch.argsort(sims, descending=True)
    
    seen_labels = set()
    top_results = []
    for idx in sorted_idx.tolist():
        lbl = infer_labels[idx]
        if lbl not in seen_labels:
            score = sims[idx].item()
            top_results.append(lbl)
            seen_labels.add(lbl)
        if len(top_results) == 10:
            break
    return top_results

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

def generate_cot(row,tokenizer_cot) :
    inputs = None
    outputs = None
    with torch.no_grad():
        inputs = tokenize_input_for_cot(row,tokenizer_cot)
        generated_outputs = model_cot.generate(
            **inputs,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id
        )
    response=""

    #them vao skip_special_tokens=True
    for i in range(inputs["input_ids"].shape[1], generated_outputs.shape[1]):
        token_id = generated_outputs[0, i]
        decoded_token = str(tokenizer.decode(token_id))
        response += decoded_token
    return response

def infer(row,category,misconception,tokenizer,thought,decode_steps):
    inputs = None
    outputs = None
    scores = -300

    print(f"---{decode_steps} turn.---")
    inputs = tokenize_input(row,category,misconception,tokenizer,thought)
    with torch.no_grad():
        outputs = model(**inputs)
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
    print(f"---Scores calculated ---")
    if decode_steps < 3:
        print(f"--- Scores = {scores} ---")
    decode_steps+=1
    result = str(category+":"+misconception)
    return scores , result , decode_steps
    
def predict(dataset,decode_steps):
    count = 0
    for row in dataset :
        print(f"--This is the {count} test---")
        candidate_labels = retrieve(row,tokenizer_retrieve,model_retrieve)
        ranking_candidate = {}
        for label in candidate_labels :
            category,misconception = None,None
            parts = label.split(":")
            category = str(parts[0])
            misconception = str(parts[1])
                
            thought = generate_cot(row,tokenizer_cot)
            score , result , decode_steps = infer(row,category,misconception,tokenizer,thought,decode_steps)
            ranking_candidate[result] = score
                
        sorted_candidate_desc = sorted(ranking_candidate.items(), key=lambda x: x[1], reverse=True)
        top_3_candidates = sorted_candidate_desc[:3]
        top_3_keys = [k for k, v in top_3_candidates]
        row['Category:Misconception'] = str(top_3_keys[0]+" "+top_3_keys[1]+" "+top_3_keys[2])
        for column_name in remove_columns :
            row.pop(column_name,None)
        if count < 3 :
            print(row)
        count+=1

def cal_map_k(k,data_df,correct_df):
    sum = 0
    count = 0
    for idx, row in data_df.iterrows() :
        for i in range(k):
            if row["Category"] == correct_df.loc[idx]["Category"] and row["Misconception"] == data_df.loc[idx] :
                sum +=  (k-i)/k
                break
        count +=1
    sum = sum/count

if __name__ == "__main__":
    
    state_dict = torch.load("best_model_proposed_110.pth", map_location="cuda")
    model_retrieve_path = 'mathbert_model/transformers/default/1'
    tokenizer_retrieve_path = 'mathbert_tokenizer/transformers/default/1'
    tokenizer_retrieve = AutoTokenizer.from_pretrained(tokenizer_retrieve_path)
    model_retrieve = AutoModel.from_pretrained(model_retrieve_path).to("cuda")
    model_retrieve = torch.nn.DataParallel(model_retrieve)
    model_retrieve.load_state_dict(state_dict)
    
    test= prepare_df1("test.csv")
    test_dataset = Dataset.from_pandas(test)

    embedded_training_dataset_path = 'embeddings_test190.csv'
    infer_data = pd.read_csv(embedded_training_dataset_path)
    
    infer_embeds_list = infer_data['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
    infer_embeds = torch.stack(list(infer_embeds_list.values))
    infer_embeds = F.normalize(infer_embeds, dim=1).to(device)
    infer_labels = infer_data['label']

    MODEL_PATH= "qwen-3/transformers/8b/1"
    # TRAIN_CSV_PATH = "/train_with_misconceptions.csv" 

    MODEL_COT ="qwen-3/transformers/1.7b/1"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    tokenizer_cot = AutoTokenizer.from_pretrained(MODEL_COT)
    if tokenizer_cot.pad_token is None:
        tokenizer_cot.pad_token = tokenizer_cot.eos_token
    model_cot = AutoModelForCausalLM.from_pretrained(
    MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
    )

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    yes_g_token_id = tokenizer.encode(" Yes",add_special_tokens=False)[0]
    no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    decode_steps=0

    remove_columns = ["QuestionId","QuestionText","MC_Answer","StudentExplanation"]

    test_dataset= predict(test_dataset,decode_steps=decode_steps)
    submit_df = pd.DataFrame(test_dataset)

    correct_df = "valid_df"
    correct_dataset = Dataset.from_pandas(correct_df)
    map_k_score = cal_map_k(k,test_dataset, correct_dataset)
    print(map_k_score)
