from transformers import Trainer, AutoModelForCausalLM ,AutoTokenizer
import torch.nn as nn 
import torch

class CrossEntropyTrainer(Trainer):
    def __init__(self, *args, yes_token_id, no_token_id,decode_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.loss_fct = nn.CrossEntropyLoss()
    
    # override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits

        last_token_logits = logits[:, -1, :]

        # yes_scores = last_token_logits[:, self.yes_token_id]
        # no_scores = last_token_logits[:, self.no_token_id]
        # scores = yes_scores - no_scores

        
        num_positives = self.args.per_device_train_batch_size
        num_negatives = num_positives * num_positives
        expected_total_size = num_positives + num_negatives

        if last_token_logits.shape[0] != expected_total_size:
            print(f"Batch size không khớp! Got {last_token_logits.shape[0]}, expected {expected_total_size}. Skipping loss calculation for this batch.")
            #khong hoc / update weight
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        positive_labels = torch.full((num_positives,), self.yes_token_id, dtype=torch.long, device=model.device)
        negative_labels = torch.full((num_negatives,), self.no_token_id, dtype=torch.long, device=model.device)
        labels = torch.cat([positive_labels, negative_labels])



        if decode_steps < 5:
            with torch.no_grad():
                gen = model.generate(
                    **{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")},
                    max_new_tokens=5000,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    do_sample=True,
                    return_dict_in_generate=False
                )
                decoded_inp = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
                decoded_gen = self.tokenizer.decode(gen[0], skip_special_tokens=False)
                print("================= DEBUG=================")
                print(f"--- INPUT ---:\n{decoded_inp}")
                print(f"--- Response ---:\n{decoded_gen}")
                print("============================================")
            decode_steps+=1



        #ce loss
        loss = self.loss_fct(last_token_logits, labels)
        return (loss, outputs) if return_outputs else loss