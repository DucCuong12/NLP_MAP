from transformers import AutoModel, AutoTokenizer
import os
print(os.chdir('..'))
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained('./stageA_mlm')
text = "This is a test sentence for the encoder."
enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
encoded_input = encoder(**enc)
print(encoded_input.last_hidden_state[:,0,:])
print(encoded_input)
