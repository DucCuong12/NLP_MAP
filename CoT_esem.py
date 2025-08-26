from pydantic import BaseModel, Field
from typing import List, Optional
import os
import openai
from dotenv import load_dotenv
import asyncio
from process_data import prepare_df, prepare_data
from datasets import Dataset
from textwrap import dedent
from typing import Union
import json
import numpy as np
from pathlib import Path
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
print(os.getenv("OPENAI_BASE_URL"))
# os.chdir('/bigdisk/cuongvd17/Testing/kaggle')
os.chdir('E:/LLM/MAP/NLP_MAP')
df = prepare_df('train.csv')
dataset = Dataset.from_pandas(df)
dataset_new = dataset.map(prepare_data, remove_columns=dataset.column_names)
example = dataset[123]
model = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))
addition_prompt = dedent(
    """         
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
)


class Score(BaseModel):
    id: int = Field(description = 'Index of response')
    score: float = Field(description= "Score of the response")
    reason: str = Field(description = 'reason of this score')
class ScoreResponse(BaseModel):
    scores: List[Score] = Field(description = 'The score for each tranjactory')
def get_response(model, example,epoch =3):
    """
    Simulate an asynchronous function that fetches responses from an LLM.
    In a real-world scenario, this would involve making an API call to the LLM service.
    """
    final_result = []
    os.chdir('E:/LLM/MAP/NLP_MAP')
    prompt = open('prompt/teacher_COT.txt', 'r').read().format(question = example['QuestionText'], answer = example['MC_Answer'], explain = example['StudentExplanation'], label = example['text_label'])
    print(prompt)
    message = [
        {"role": "system",  "content": "You are a professional mathematician, educator, and evaluator. You are tasked with evaluating the correctness of a student's answer to a math question. Your evaluation should be based on the provided rubric and should consider the student's explanation, the question, and the answer."
        },
        {"role": "user", "content": prompt}
        ]
    list_model = ['gpt-4.1-nano']
    for _ in range(epoch):
        idx = np.random.randint(0, len(list_model))
        responses = model.chat.completions.create(
            model = 'gpt-4.1-nano',
            messages = message,
            max_tokens = 256, 
            )
        response = responses.choices[0].message.content
        final_result.append(response)
    return final_result
async def get_score(
    responses: List[str], 
    additional_prompt: str,
    model 
    ) -> ScoreResponse:
    system_prompt = open('prompt/goal_CoT.txt').read().format(rubric = additional_prompt)
    user_response = open('prompt/score_COT.txt').read().format(
        question = example['QuestionText'],
        answer = example['MC_Answer'],
        explain = example['StudentExplanation'],
        label = example['text_label']
    )
    for idx, response in enumerate(responses):
        user_response+= f'\n<trajectory id= "{idx}"\n{response}\n</tranjectory>\n"'
    responses = model.chat.completions.parse(
        model = 'gpt-4.1-nano',
        messages = [
            {'role': "system", "content": system_prompt},
            {'role': "user", "content": user_response}
        ],
        response_format =  ScoreResponse,
        max_tokens = 256,   
    )
    return json.loads(responses.choices[0].message.content)['scores']
    
    
    
    
if __name__ == '__main__':
    model = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'), base_url="https://llm-prof-tien.thaiminhpv.id.vn")
    print(os.getenv('OPENAI_API_KEY'))
    print(os.getenv("OPENAI_BASE_URL"))
    sampled = dataset.shuffle(seed=42).select(range(10))
    prompts = []
    scores = []
    for example in sampled:

        responses = get_response(model, example)
        score = asyncio.run(get_score(responses,addition_prompt, model))
        prompts.append(responses)
        scores.append(score)

    results = [{"prompts": p, "score": s} for p, s in zip(prompts, scores)]
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)