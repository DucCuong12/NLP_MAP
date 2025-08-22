import torch
import argparse
import gc
import os
import random
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import json
import http
import time
from openai import OpenAI

label = "HIDDEN"
save_dir = os.path.join(os.getcwd(), "generated_results")
os.makedirs(save_dir, exist_ok=True) 
# random seed
seed = random.randint(100, 10000)   
print(f"setting seed to: {seed}")
random.seed(seed)

df = pd.read_csv("../train.csv")

ds = Dataset.from_pandas(df)
question_id = ds["QuestionId"]

# model_LLM = "Model_LLM"
load_dotenv()
model_LLM = "gpt-4.1-nano"
api_key = os.getenv("API_KEY_LLM")
api_endpoint = "llm-prof-tien.thaiminhpv.id.vn"

def get_response_2(prompt_content, temperature=0.7, top_p=0.8, max_tokens=512):
    client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    )
    completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt_content
        }
    ],
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    )  
    return completion.choices[0].message.content



def get_response(prompt_content, temperature=0.7, top_p=0.8, max_tokens=512):
    payload_explanation = json.dumps(
        {
            "model": model_LLM,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
    )

    headers = {
        "Authorization": "Bearer " + api_key,
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "x-api2d-no-cache": "1",
    }

    response = None
    n_trial = 0
    while n_trial < 2:
        n_trial += 1
        try:
            conn = http.client.HTTPSConnection(api_endpoint)
            conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
            res = conn.getresponse()
            data = res.read()

            json_data = json.loads(data)

            if "choices" in json_data and json_data["choices"]:
                response = json_data["choices"][0]["message"]["content"]
            else:
                print("Unexpected API response:", json_data)
                response = None
            break
        except Exception as e:
            print(f"Error in API (trial {n_trial}): {e}")
            time.sleep(1)
            continue

    return response


# def generate_text(prompt, max_tokens=384, temperature=0.7, top_p=0.8):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         repetition_penalty=1.0,
#         do_sample=True,
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def generate_text(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=384, temperature=0.7, top_p=0.8):
#     response = openai.ChatCompletion.create(
#     model=model,
#     messages=[
#         {"role": "user", "content": prompt},
#     ],
#     temperature=temperature,
#     top_p=top_p,
#     max_tokens=max_tokens,
#     )
#     return response.choices[0].message.content


# sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.0, max_tokens=384)

prompts = []

for example in ds:
    question = example["QuestionText"]
    student_answer = example["MC_Answer"]
    student_explan = example["StudentExplanation"]
    cat = example.get("Category", "")
    mis = example.get("Misconception", "")
    if pd.isna(mis):
        mis = "NA"
    label = cat + ":"+ mis
    user_message = f"Question: {question}\nStudent's Answer: {student_answer}\nStudent's explanation: {student_explan}"

    sp = f"""
        You are a meticulous educational analyst and expert in mathematics pedagogy. 
        Your task is to verify whether a proposed classification of a student's response to a math problem is entirely accurate.

        The label format is A_B:C, where:
        - A: True if the student's answer is correct solution to the question, False if incorrect.
        - B: Misconception if the student's explanation has a misconception; Correct if no misconception; Neither if the student explanation is incorrect, but does not point to a specific misconception, it could be a guess, irrelevant, or simply nonsensical.
        - C: If B is Misconception, specify the type of misconception; if B is not Misconception, C is NA.

        You are given the correct label: {label}, but during your reasoning, assume you do not know this label until the final step.

        Analyze the student's answer and explanation step-by-step, without using the given label in your reasoning.  
        At the end, give your final prediction in the format A_B:C.

        YOUR STEP-BY-STEP VERIFICATION PROCESS (Chain-of-Thought):
        1. Reasoning about A: Is the student's answer correct or incorrect?
        2. Reasoning about B: Does the student have a misconception, or is the answer correct or neither?
        3. If B is Misconception, identify the type; otherwise, write NA.
        4. Explain the student's reasoning.
        5. Final prediction: Provide your label in the format A_B:C.

        Keep your entire answer under 512 words.
    """
    text = f"{sp}\n\nQuery: {user_message}\nAnswer:\n"
    prompts.append(text)


chunk_size = 16
max_new_tokens = 384
# print(tokenizer.padding_side)  
# print(tokenizer.pad_token, tokenizer.pad_token_id)
# tokenizer.padding_side = "left"
with torch.inference_mode():
    for i in range(0, len(ds), chunk_size):
        batch_prompts = prompts[i:i + chunk_size]
        batch_question_id = question_id[i:i + chunk_size]

        # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     temperature=0.7,
        #     top_p=0.8,
        #     repetition_penalty=1.0,
        #     do_sample=True
        # )

        
        result = []
        generated_texts = []
        all_prompts = []
        for p in batch_prompts:
            generation_text = get_response_2(p, max_tokens=1024)
            all_prompts.append(p)
            generated_texts.append(generation_text)
            full_texts = f"{p}{generation_text}"
            result.append(full_texts)  
        # for j in range(len(batch_prompts)):
        #     input_len = inputs["input_ids"][j].shape[0]
        #     gen_ids = outputs[j][input_len:]
        #     gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        #     generated_texts.append(gen_text)

        df_chunk = pd.DataFrame({
            "query_id": batch_question_id,
            "prompt": all_prompts,
            "generated_text": generated_texts,
            "full_text": result
        })

        intermediate_path = os.path.join(save_dir, f"generated_{seed}_chunk_{i // chunk_size}.parquet")
        df_chunk.to_parquet(intermediate_path)
        print(f"Saved chunk {i // chunk_size} to {intermediate_path}")