import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from process_data import prepare_df
import os
from fine_tune1 import infer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def retrieve_all(
    test_dataset,
    tokenizer,
    model,
    num_of_top_result,
    infer_embeds,
    infer_labels,
    batch_size=32
):
    texts = test_dataset['text_train']  
    total = len(texts)

    all_candidate_labels = []
    all_candidate_scores = []
    model.eval()
    test_data = DataLoader(texts, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    for batch in tqdm(test_data, desc = "Retrieving candidates"):
        tokenized = tokenizer(
            batch,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        ).to(device = 'cuda')

        with torch.no_grad():
            output = model(**tokenized)
        
        # [B, H]
        queries = output.last_hidden_state[:, 0, :]
        queries = F.normalize(queries, dim=1).to(device = 'cuda')                

        sims_batch = torch.matmul(queries, infer_embeds.T)    

        for row_idx in range(sims_batch.size(0)):
            sims = sims_batch[row_idx]
            sorted_idx = torch.argsort(sims, descending=True)

            seen_labels = set()
            top_results = []
            top_scores = []
            for cand_idx in sorted_idx.tolist():
                lbl = infer_labels[cand_idx]
                if lbl not in seen_labels:
                    top_results.append(lbl)
                    top_scores.append(sims[cand_idx].item())
                    seen_labels.add(lbl)
                if len(top_results) == num_of_top_result:
                    break

            all_candidate_labels.append(top_results)
            all_candidate_scores.append(top_scores)
    test_dataset.add_column("candidate_labels",all_candidate_labels)
    test_dataset.add_column("candidate_label_scores",all_candidate_scores)

    return test_dataset,all_candidate_labels, all_candidate_scores

def cal_map_ks(predictions_per_row, correct_labels, ks):
   
    # Normalize ks
    if not ks:
        return {}

    if hasattr(correct_labels, "reset_index"):
        correct_labels = correct_labels.reset_index(drop=True)
    if hasattr(correct_labels, "tolist"):
        correct_labels = correct_labels.tolist()
    n = len(predictions_per_row)
    if n == 0:
        return {k: 0.0 for k in ks}

    max_k = ks[-1]
    sums = {k: 0.0 for k in ks}

    for i in range(n):
        preds = predictions_per_row[i]
        true_label = correct_labels[i]
        truncated = preds[:max_k]
        match_rank = None
        for r, lbl in enumerate(truncated):
            if lbl == true_label:
                match_rank = r
                break
        if match_rank is not None:
            for k in ks:
                if match_rank < k:
                    sums[k] += 1.0 / (match_rank + 1)

    return {k: (sums[k] / n) for k in ks}
def infer_retrieval_only(
    all_candidate_labels,
    correct_labels,
    ks
):

    map_k_scores_retrieval = cal_map_ks(all_candidate_labels,correct_labels,ks)
    print(str("MAP@K scores for our proposed model:\n" + str(map_k_scores_retrieval)))
    with open("map_scores.txt","a") as f:
        f.write("MAP@K scores for our proposed model"+str(map_k_scores_retrieval) + "\n")
if __name__ == "__main__":
    model_name ='tbs17/MathBERT'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    print(os.getcwd())
    state_dict = torch.load('checkpoints/new_50_120.pth')
    model.load_state_dict(state_dict)
    df = prepare_df('train_split.csv')
    dataset = Dataset.from_pandas(df)
    def tok(row):
        text = row['text_train']
        return tokenizer(text, truncation=True, padding='max_length', max_length=512)

    dataset = dataset.map(tok, batched=True, remove_columns=[col for col in dataset.column_names if col not in ['text_label', 'label']])



    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        labels = [item["text_label"] for item in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }

    dataset.set_format(type="torch")  # hoặc dataset = dataset.with_format("torch")
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    text_result = infer(model, train_dataloader)
    # print(text_result)

    infer_embeds = torch.stack([x['text'] for x in text_result]).to(device = 'cuda')  # shape: (num_samples, hidden_size)

    infer_labels = [x['label'] for x in text_result]
    test_dataset = prepare_df('split_test.csv')
    test_dataset = Dataset.from_pandas(test_dataset)
    correct_labels = test_dataset['text_label']
    test_dataset,all_candidate_labels, all_candidate_scores = retrieve_all(test_dataset,tokenizer,model,10,infer_embeds,infer_labels,batch_size=32)
    print(all_candidate_labels[0])
    infer_retrieval_only(all_candidate_labels,correct_labels,[1,3,5])