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
{thoughts[i]}

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


