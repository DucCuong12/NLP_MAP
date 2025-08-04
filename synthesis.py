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
from pathlib import Path
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
print(os.getenv("OPENAI_BASE_URL"))
os.chdir('/bigdisk/cuongvd17/Testing/kaggle')
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
class LLMResponse(BaseModel):
    # id: str = Field(description="The ID of the response.")
    question: str = Field(description="The response question.")
    answer: Optional[str] = Field(description = "The answer choose")
    explain: str = Field(description = 'The explanation of student for this question')
    label: str = Field(description = 'Label of example')

class Score(BaseModel):
    id: int = Field(description = 'Index of response')
    score: float = Field(description= "Score of the response")
    reason: str = Field(description = 'reason of this score')
class ScoreResponse(BaseModel):
    scores: List[Score] = Field(description = 'The score for each tranjactory')
def get_response(model, example,epoch =3) -> List[LLMResponse]:
    """
    Simulate an asynchronous function that fetches responses from an LLM.
    In a real-world scenario, this would involve making an API call to the LLM service.
    """
    template_append = "Generate a new question with a distinct explanation strategy—different from any previously generated—to explore a novel perspective on the same concept or misconception."
    final_result = []
    os.chdir('/bigdisk/cuongvd17/Testing/kaggle/NLP_MAP')
    prompt = open('prompt/generate.txt').read().format(answer = example['MC_Answer'], question = example['QuestionText'], explain = example['StudentExplanation'], label = example['text_label'])
    print(prompt)
    message = [
        {"role": "system",  "content": "You are a distinguished mathematician and educator. Given an example question, student answer, and explanation, your task is to craft a single, original question with its corresponding answer choice and student explanation that illustrates the same underlying concept or misconception."
        },
        {"role": "user", "content": prompt}
        ]
    for _ in range(epoch):
        responses = model.chat.completions.parse(
            model = 'deepseek-v3-0324',
            messages = message,
            response_format = LLMResponse,
            max_tokens = 256, 
            )
        message.append(
            {"role": "assistant", "content": responses.choices[0].message.content }
        )
        message.append(
            {"role": "user", 'content': template_append}
        )
        response = json.loads(responses.choices[0].message.content)
        final_result.append(LLMResponse(**response))
    return final_result
async def get_score(
    responses: List[LLMResponse], 
    additional_prompt: str,
    model 
    ) -> ScoreResponse:
    system_prompt = open('prompt/goal.txt').read().format(rubric = additional_prompt)
    user_response = ''
    for idx, response in enumerate(responses):
        user_response+= f'<trajectory id= "{idx}"\n{response}\n</tranjectory>\n"'
    responses = model.chat.completions.parse(
        model = 'deepseek-v3-0324',
        messages = [
            {'role': "system", "content": system_prompt},
            {'role': "user", "content": user_response}
        ],
        response_format =  ScoreResponse,
        max_tokens = 2048,   
    )
    return json.loads(responses.choices[0].message.content)['scores']
    
    
    
    
if __name__ == '__main__':
    responses = get_response(model, example)
    scores = asyncio.run(get_score(responses,addition_prompt, model))
    print(scores)