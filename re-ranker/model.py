from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn 
import torch

class CrossEntropyTrainer(Trainer):
    def __init__(self, *args, yes_g_token_id, no_g_token_id, yes_token_id, no_token_id, tokenizer, decode_steps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.decode_steps = decode_steps
        self.tokenizer = tokenizer
        self.yes_g_token_id = yes_g_token_id
        self.no_g_token_id = no_g_token_id
        self.loss_fct = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        label = inputs.pop("label")
        
        outputs = model(**inputs)
        logits = outputs.logits  
        
        last_token_logits = logits[:, -1, :]  
        
        yes_logits = torch.max(
            last_token_logits[:, self.yes_token_id],
            last_token_logits[:, self.yes_g_token_id]
        )
        no_logits = torch.max(
            last_token_logits[:, self.no_token_id],
            last_token_logits[:, self.no_g_token_id]
        )
        # theo chieu doc --> gom 2 tensor list : 1 toan no, 1 toan yes
        binary_logits = torch.stack([no_logits, yes_logits], dim=1)
        
        # 1 ="Yes" /" Yes" ; 0 = "No" / " No"
        binary_labels = (label == 1).long()
        
        if self.decode_steps < 3:
            with torch.no_grad():
                debug_outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
                for i, seq in enumerate(debug_outputs):
                    decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                    print(f"Batch {i} generated: {decoded}") 
            
            print(f"--- Labels: {label} ---")
            print(f"--- Max Yes Logits: {yes_logits} ---")
            print(f"--- Max No Logits: {no_logits} ---")
            self.decode_steps += 1
        
        loss = self.loss_fct(binary_logits, binary_labels)
        
        inputs["label"] = label
        
        return (loss, outputs) if return_outputs else loss

