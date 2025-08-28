from openai import embeddings
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
import argparse
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

######### Call model ##########
model_name = 'tbs17/MathBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)  
# os.chdir('/bigdisk/cuongvd17/Testing/kaggle')
print(os.getcwd())
model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
# state_dict = torch.load('checkpoints/best_model_proposed_190.pth')  
# model.load_state_dict(state_dict)  # Use strict=False to ignore missing keys
# model.load_state_dict(torch.load("checkpoints/best_model_proposed.pth"))
######## Get data #############
df = prepare_df('train.csv')
dataset = Dataset.from_pandas(df)
print(dataset)

### Tokenizer data input ######
def tok(row):
    text = row['text_train']
    return tokenizer(text, truncation=True, padding='max_length', max_length=512)

dataset = dataset.map(tok, batched=True, remove_columns=[col for col in dataset.column_names if col not in ['text_label', 'label']])



def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    labels = [item["text_label"].replace(':', '_').split("_") for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

dataset.set_format(type="torch")  # hoặc dataset = dataset.with_format("torch")
######## Training ##############
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, drop_last=True, collate_fn=collate_fn)


def similarity(label_i, label_j):
    if label_i == label_j:
        return 1.0
    elif label_i[0] == label_j[0]:
        if label_i[1] == label_j[1]:
            return 0.7
        else:
            return 0.4
    elif label_i[1] == label_j[1]:
        if label_i[2] == label_j[2]:
            return 0.3
        else:
            return 0.1
    # elif label_i[2] == label_j[2]:
    #     return 0.1
    else:
        return 0

def build_mask(labels):
    batch = len(labels)
    mask = torch.zeros((batch, batch), dtype=torch.float32, device="cuda")

    for i in range(batch):
        for j in range(batch):
            if i == j:
                sim =0
            else:
                sim = similarity(labels[i], labels[j])
            mask[i][j] = sim
    
    return mask

def contrastive_loss_2(outputs, labels, temperature = 0.09):
    mask = build_mask(labels)
    embed = F.normalize(outputs, dim=1)
    similarity_matrix  = torch.matmul(embed, embed.T) / temperature  

    exp_sim = torch.exp(similarity_matrix) * (~torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool())
    similarity_mat = (similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)) * mask
    mean_log = (similarity_mat.sum(dim=1) / (mask.sum(dim=1) + 1e-8))
    return -mean_log.mean()





def contrastive_loss(outputs, labels,temperature = 0.09):
    # Compute the contrastive loss between the outputs and labels
    embed = F.normalize(outputs, dim=1)
    similarity_matrix  = torch.matmul(embed, embed.T) / temperature  
    
    pos_mask = (labels == labels.T) & (~torch.eye(labels.shape[0], device=labels.device).bool())
    exp_sim = torch.exp(similarity_matrix) * (~torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool())
    similarity_mat = (similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)) * pos_mask
    mean_log = (similarity_mat.sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8))
    return -mean_log.mean()
    

def fine_tune(model, train_dataloader, arg, epochs = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
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
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            # labels = batch['label'].contiguous().view(-1, 1).to(device)
            labels = batch["labels"]
            with autocast('cuda'):
                outputs = model(input_ids, attention_mask, token_type_ids)
                outputs = outputs.last_hidden_state[:,0,:]
                loss = contrastive_loss_2(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
        output_dir = "checkpoints"
        os.makedirs(output_dir, exist_ok=True)
        if arg.loc:
            with open('new', 'a') as f:
                f.write(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}\n')
        else:
            with open('continue', 'a') as f:
                f.write(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}\n')
        if epoch % 20 == 0:
            if arg.loc:
                torch.save(model.state_dict(), f"checkpoints/new_loss_{epoch}.pth")
                print(f"Saved best model at epoch {epoch+1} with loss {total_loss:.4f}")
            else:
                torch.save(model.state_dict(), f"checkpoints/continue_loss_{epoch}.pth")
                print(f"Saved best model at epoch {epoch+1} with loss {total_loss:.4f}")
def infer(model,train_dataloader):
    text = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    model.to(device)
    model.eval()
    for batch in tqdm(train_dataloader, desc="Inference"):
        with torch.no_grad():
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            label = batch['text_label']
            for i in range(len(label)):
                text.append({'text': outputs.last_hidden_state[:,0,:][i].detach().cpu(), 'label': label[i]})
    return text
def save_embeddings_to_csv(text, filename="embeddings_loss_test240.csv"):
    rows = []
    for item in text:
        emb = item['text'].cpu().numpy().tolist()
        label = item['label']
        # Nếu label là tensor, chuyển sang python type
        if torch.is_tensor(label):
            label = label.item() if hasattr(label, 'item') else label
        # Lưu embedding dưới dạng chuỗi
        rows.append({'text_embed': str(emb), 'label': label})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved embeddings to {filename}")

def args():
    parser = argparse.ArgumentParser(description="Training script for multimodal model")
    parser.add_argument('--mode', type=str, default='train', help='train or infer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--pro', action='store_true', help='continue training')
    parser.add_argument('--loc', action='store_true', help='local machine')
    return parser.parse_args()
if __name__ == "__main__":
    args = args()
    if args.mode == 'train':
        print("Starting training...")
        if args.pro:
            os.chdir('/bigdisk/cuongvd17/Testing/kaggle')

            print("Continue training from checkpoints/best_model_proposed_110.pth")
            state_dict = torch.load('checkpoints/best_model_proposed_110.pth')
            model.load_state_dict(state_dict)  # Use strict=False to ignore missing keys
        else:
            print("Starting training from scratch.")
            # state_dict = torch.load('checkpoints/new_loss_100.pth')
            # model.load_state_dict(state_dict)
        fine_tune(model, train_dataloader, args, epochs=args.epochs)
        print("Training complete. Model saved to checkpoints/best_model_proposed.pth")
    else:
        print("Starting inference...")
        os.chdir('/bigdisk/cuongvd17/Testing/kaggle/NLP_MAP')
        state_dict = torch.load('checkpoints/new_loss_240.pth')
        model.load_state_dict(state_dict)
        train_dataloader1 = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

        text = infer(model, train_dataloader1)
        save_embeddings_to_csv(text)
        print("Inference complete.")