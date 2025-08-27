from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd


class TokenizeData :
    def __init__(self,df):
        self.df = df 
    def preprocess_function_mini_batch(self,thoughts):
        df = self.df
        prompts = []
        for q_text, mc_answer, explanation, category, misconception , i in zip(
            df['QuestionText'],
            df['MC_Answer'],
            df['StudentExplanation'],
            df['Category'],
            df['Misconception'],
            range(len(df['StudentExplanation']))

        ):
            
            prompt = f"""
<|im_start|>system
Response in maximum 256 words.
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem, and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the evidence.
DEFINITIONS OF THE CLASSIFICATION LABELS:
category: This is a compound label with two parts, separated by an underscore: Correctness_ReasoningType.

Part 1: Correctness (True or False): This describes whether the student's mc_answer is objectively the correct solution to the q_text.

Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
Correct: The explanation shows sound, logical, and mathematically valid reasoning.
Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

Part 3: Misconception : This is a text description of the specific thinking error. It is only relevant when the ReasoningType in the category is Misconception. If the category is ..._Correct or ..._Neither, this field's value should be "NA"

THOUGHT ANALYSIS:
{thoughts[i]}

YOUR TASK:
1. Compare the Thought Analysis to the Proposed Classification (Category + Misconception) in Problem Data, then answer the following 3 questions :
1.1 From the thought analysis and your analysis, does the your True/False conclusion of student's answer match the first part of the {category} label?
1.2 Does your Correct/Misconception/Neither conclusion from Step 2 match the second part of the {category} label?
1.3 If the category is ..._Misconception, does the student's error you identified align with the provided {misconception} text? (If the category in Proposed Classification is ..._Correct or ..._Neither, you can skip this step) 
2. Final Conclusion: A "Yes" is only possible if all checks in Step 1 pass. If there is any mismatch at any point, the answer must be "No". Give the "Yes/No" word conclusion at a newline with no tokens/words after it (even a dot "."), this is the end of your response.

Show your detailed reasoning by following these steps

<|im_end|>
<|im_start|>user
Problem Data:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

Proposed Classification:
Category: '{category}'
Misconception: '{misconception}'

<|im_end|>
<|im_start|>assistant
"""
            prompts.append(prompt)
        
        df['prompt'] = prompts
        # model_inputs = tokenizer(prompts, max_length=1024, truncation=True)
        # return model_inputs
        return df

    def tokenize_function(self,examples,tokenizer):
        tx = tokenizer(
            examples['prompt'],
            padding=False,
            truncation=True,
            max_length=2048,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def get_dataset(self,thoughts,tokenizer):
        df = self.df
        df= self.preprocess_function_mini_batch(thoughts)
        self.df = df 
        dataset = Dataset.from_pandas(df)
        remove_columns = ["QuestionId","QuestionText","MC_Answer","StudentExplanation","Category","Misconception","prompt",]
        task_dataset = dataset.map(self.tokenize_function,batched=False,remove_columns=remove_columns,fn_kwargs = {'tokenizer' : tokenizer})
        return task_dataset

