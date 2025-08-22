IGNORE_INDEX = -100
from transformers import AutoTokenizer
from copy import deepcopy
from datasets import Dataset
def find_token(input_ids, token_pattern):
    ret = []
    token_pattern_len = len(token_pattern)
    for ex_input_ids in input_ids:
        search_end = len(ex_input_ids)
        found = False
        for j in range(search_end - token_pattern_len, -1, -1):
            if ex_input_ids[j : j + token_pattern_len] == token_pattern:
                ret.append(j + token_pattern_len)
                found = True
                break
        if not found:
            ret.append(0) 
    return ret
    


class MathDataset:
    
    def __init__(self, model_name, max_length = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        self.token_pattern = self.tokenizer.encode("Answer:\n", add_special_tokens=False)
        self.eos_token = self.tokenizer.eos_token
        self.max_length = max_length
    def pre_process(self, df):
        formatted_texts = []
        system = "Analyze the student's answer to determine whether it is correct or incorrect. If incorrect, detect flaws in the student's reasoning."

        for _, row in df.iterrows():
            question = row["QuestionText"]
            # correct_answer = row["Category"].split("_")[0]
            mc_answer = row["MC_Answer"]
            student_explanation = row["StudentExplanation"]
            user_message = f"Question: {question}\nStudent Answer: {mc_answer}\nStudent Explanation: {student_explanation}"
            assistant_llm = row["generated_text"]           
            text = f"{system}\n\nQuery: {user_message}\nAnswer:\n{assistant_llm}{self.eos_token}"
            formatted_texts.append(text)
            
        df["text_train"] = formatted_texts
        
        return df

    def tokenize(self, examples):
        tokenized = self.tokenizer(
            examples["text_train"],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_length=True,
        )

        labels = deepcopy(tokenized["input_ids"])
        start_index = find_token(tokenized["input_ids"], self.token_pattern)
        for idx, src_len in enumerate(start_index):
            labels[idx][:src_len] = [IGNORE_INDEX] * src_len
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "length": tokenized["length"],
            "label": labels
        }

    def get_dataset(self, df):
        df = df.copy()
        df = self.pre_process(df).reset_index(drop=True)

        ds = Dataset.from_pandas(df)
        ds = ds.map(self.tokenize, batched=True)

        return ds
        