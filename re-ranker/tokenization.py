from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd


class TokenizeData :
    def __init__(self,positive_df_path,negative_df_path):
        self.positive_df_path = positive_df_path
        self.negative_df_path = negative_df_path
        self.positive_df = pd.read_csv(self.positive_df_path,keep_default_na=False)
        self.negative_df = pd.read_csv(self.negative_df_path,keep_default_na=False)
    def preprocess_function_mini_batch(self,df):
        prompts = []
        for q_text, mc_answer, explanation, category, misconception in zip(
            df['QuestionText'],
            df['MC_Answer'],
            df['StudentExplanation'],
            df['Category'],
            df['Misconception']
        ):
            
            prompt = f"""<|im_start|>system
    You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem, and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the evidence.
    DEFINITIONS OF THE CLASSIFICATION LABELS:
    category: This is a compound label with two parts, separated by an underscore: Correctness_ReasoningType.

    Part 1: Correctness (True or False): This describes whether the student's mc_answer is objectively the correct solution to the q_text.

    Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
    Correct: The explanation shows sound, logical, and mathematically valid reasoning.
    Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
    Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

    misconception: This is a text description of the specific thinking error. It is only relevant when the ReasoningType in the category is Misconception. If the category is ..._Correct or ..._Neither, this field's value should be "NA" or "nan" or "NAN" ,...

    YOUR STEP-BY-STEP VERIFICATION PROCESS (Chain-of-Thought):

    1. Analyze Answer Correctness (True/False Check): First, independently solve the math problem in {q_text}. Compare your result to the student's {mc_answer}. Is the student's answer objectively True (correct) or False (incorrect)?
    2. Analyze Explanation Quality (Reasoning Check): Now, ignore the final answer and focus only on the {explanation}.
    Deconstruct the student's logic. What steps did they follow?
    Based only on their text, classify their reasoning: Is it Correct, a clear Misconception, or Neither?
    If you identify a misconception, briefly describe it in your own words.
    3. Compare Your Analysis to the Provided Labels: Now, compare your findings from steps 1 and 2 with the given {category} and {misconception}.
    Does your True/False conclusion from Step 1 match the first part of the {category} label?
    Does your Correct/Misconception/Neither conclusion from Step 2 match the second part of the {category} label?
    If the category is ..._Misconception, does the student's error you identified align with the provided {misconception} text?
    4. Final Conclusion: A "Yes" is only possible if all checks in Step 3 pass. If there is any mismatch at any point, the answer must be "No".
    Show your detailed reasoning by following these steps. Then, on the very last line, provide the final answer as exactly one word: "Yes" or "No".
    <|im_end|>
    <|im_start|>user
    Problem Data:
    Question: {q_text}
    Student's Answer: {mc_answer}
    Student's Explanation: {explanation}
    Proposed Classification:
    Category: '{category}'
    Misconception: '{misconception}'
    Verification Task:
    Based on the data and the definitions provided, is the 'Proposed Classification' an accurate description of the 'Problem Data'?
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
            max_length=3000,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def get_dataset(self,df,tokenizer):
        df= self.preprocess_function_mini_batch(df)
        dataset = Dataset.from_pandas(df)
        remove_columns = ["QuestionId","QuestionText","MC_Answer","StudentExplanation","Category","Misconception","prompt",]
        task_dataset = dataset.map(self.okenize_function,batched=True,remove_columns=remove_columns,fn_kwargs = {'tokenizer' : tokenizer},batch_size=7000,num_proc=2)
        return task_dataset

