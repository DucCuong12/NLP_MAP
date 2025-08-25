import torch
import os
import pandas as pd
from datasets import Dataset
from transformers import pipeline,AutoTokenizer
from tqdm import tqdm
from openai import OpenAI
from process_data import prepare1_df
from torch.utils.data import DataLoader
save_dir = os.path.join(os.getcwd(), "generated_result")
os.makedirs(save_dir, exist_ok=True) 
os.chdir('/bigdisk/cuongvd17/Testing/kaggle')
df = prepare1_df('train.csv')

dataset = Dataset.from_pandas(df)
# Prepare text and label fields
os.chdir('/bigdisk/cuongvd17/Testing/kaggle/NLP_MAP')
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"   

model = pipeline(
    "text-generation",
    model = model_id,
    tokenizer=tokenizer,          # dùng tokenizer đã chỉnh
    torch_dtype= 'auto',
    device_map = 'auto',
    batch_size=32,
)
with open('prompt/generate_COT.txt', 'r') as f:
    prompt_template = f.read()
batch_size = 32
results = []
prompts = []
question_id = dataset['QuestionId']
for ex in dataset:
    prompt_new = prompt_template.format(
        question=ex['QuestionText'],
        answer=ex['MC_Answer'],
        explain=ex['StudentExplanation'],
        label=ex['text_label'],
        label1=ex['text_label'].split('_')[0],
        label2=ex['text_label'].split('_')[1].split(':')[0],
        label3=ex['text_label'].split('_')[1].split(':')[1],
    )
    
    prompts.append(prompt_new)
print('done')

data = DataLoader(prompts, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
# for start in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
for batch in tqdm(data, desc = "gen"):
    outputs = model(
        batch,
        temperature=0.2,
        top_p=0.8,
        max_new_tokens=256,
        return_full_text=False
    )
    # print(outputs[0])
    for out in outputs:
        results.append(out[0]['generated_text'].split('.', 1)[1].strip())

# Save
df_out = pd.DataFrame({
    "query_id": dataset['QuestionId'],
    "generated_text": results,
})
df_out.to_parquet("generated_result/teacher_outputs.parquet", index=False)
print("✅ Done!")