from transformers import AutoTokenizer, AutoModel
import os
from datasets import Dataset
from process_data import prepare_df
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

######### Call model ##########
model_name = 'tbs17/MathBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)    
# model.load_state_dict(torch.load("checkpoints/best_model_proposed.pth"))
######## Get data #############
os.chdir('/bigdisk/cuongvd17/Testing/kaggle/')
df = prepare_df('train.csv')
dataset = Dataset.from_pandas(df)
print(dataset)

### Tokenizer data input ######
def tok(row):
    text = row['text_train']
    return tokenizer(text, truncation=True, padding='max_length', max_length=512)

dataset = dataset.map(tok, batched=True, remove_columns=[col for col in dataset.column_names if col != 'label'])


dataset.set_format(type="torch")  # hoặc dataset = dataset.with_format("torch")
######## Training ##############
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

def contrastive_loss(outputs, labels,temperature = 0.09):
    # Compute the contrastive loss between the outputs and labels
    embed = F.normalize(outputs, dim=1)
    similarity_matrix  = torch.matmul(embed, embed.T) / temperature  
    
    pos_mask = (labels == labels.T) & (~torch.eye(labels.shape[0], device=labels.device).bool())
    exp_sim = torch.exp(similarity_matrix) * (~torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool())
    similarity_mat = (similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)) * pos_mask
    mean_log = (similarity_mat.sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8))
    return -mean_log.mean()
    

def fine_tune(model, train_dataloader, epochs = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    print(device)
    model.to(device)
    if hasattr(model, 'module'):
        params = model.module.parameters()
    else:
        params = model.parameters()
    optimizer = AdamW(params, lr=2e-5, weight_decay=0.01)
    scaler = GradScaler('cuda')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].contiguous().view(-1, 1).to(device)
            with autocast('cuda'):
                outputs = model(input_ids, attention_mask, token_type_ids)
                outputs = outputs.last_hidden_state[:,0,:]
                loss = contrastive_loss(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
        output_dir = "checkpoints"
        os.makedirs(output_dir, exist_ok=True)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/best_model_proposed_{epoch}.pth")
            print(f"Saved best model at epoch {epoch+1} with loss {total_loss:.4f}")

if __name__ == "__main__":
    fine_tune(model, train_dataloader, epochs=200)
    print("Training complete. Model saved to checkpoints/best_model_proposed.pth")