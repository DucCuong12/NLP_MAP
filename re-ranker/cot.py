from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

class CotModel:
    def __init__(self,model_cot_name,quantization_config =None):
        self.model_cot_name = model_cot_name
        self.quantization_config = quantization_config
        self.model_cot =  AutoModelForCausalLM.from_pretrained(
            model_cot_name, device_map="auto", trust_remote_code=True, quantization_config =quantization_config,
        )
        self.tokenizer_cot = AutoTokenizer.from_pretrained(model_cot_name)
        if self.tokenizer_cot.pad_token is None:
            self.tokenizer_cot.pad_token = self.tokenizer_cot.eos_token
        
    def tokenize_input_for_cot(self,df):
        tokenizer = self.tokenizer_cot
        texts = []
        for idx,row in df.iterrows():
            q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
            prompt = f"""
    <|im_start|>system
    Response in maximum 256 words.
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
            texts.append(text)
        tokenized_output = tokenizer(
            texts,
            truncation = True,
            max_length = 1024,
            return_tensors = "pt"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tokenized_output.to(device)
    
    def generate_cot(self,queries,batch_size = 2) :

        tokenizer = self.tokenizer_cot
        model_cot = self.model_cot
        thoughts = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
        
            with torch.no_grad():
                generated_outputs = model_cot.generate(
                    **batch,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id
                )
            input_length = batch["input_ids"].shape[1]  
            for i in range(batch_size):
                new_tokens = generated_outputs[i, input_length:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                thoughts.append(response)
        return thoughts