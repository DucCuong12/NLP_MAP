from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import json

def add_special_tokens(label):
    return f"[CLS] {label} [SEP]"


def extract_answer(df, question_id):
    return df.loc[df.QuestionId == question_id].MC_Answer.unique().tolist()

def format_options(options):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Extend if needed
    return ' '.join([f"{letters[i]}:{opt}" for i, opt in enumerate(options)])

def prepare_df(path):
    # path = args.path
    df = pd.read_csv(path)
    le = LabelEncoder()
    df.Misconception = df.Misconception.fillna('NA')
    df['text_label'] = df.Category + ':' + df.Misconception
    df['label'] = le.fit_transform(df['text_label'])    
    idx = df.apply(lambda row: row['Category'].split('_')[0], axis=1) == 'True'
    correct = df.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates('QuestionId')
    correct = correct[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1   
    df = df.merge(correct, on=['QuestionId', 'MC_Answer'], how = 'left')
    df.is_correct = df.is_correct.fillna(0)
    df['options'] = df['QuestionId'].apply(lambda x: extract_answer(df, x))
    print(df.iloc[0])
    df['text_train'] = df.apply(
    lambda row: f"[CLS] Question: {row['QuestionText']}\n [SEP] List of Answer: {format_options(row['options'])}\n [SEP] Choosen answer: {row['MC_Answer']}\n [SEP] Student explanation: {row['StudentExplanation']}\n [SEP]",
    axis=1
    )
    return df

def prepare_data(example):
    return {
        'text': example['text_train'],
        'label': example['label'],
        'miss': example['Misconception'],
        'is_correct': example['is_correct']
    }


def create_data_and_save(df, num_negative=15, out_file="formatted_data.json"):
    labels = sorted(df['text_label'].unique().tolist())
    labels_with_tokens = [add_special_tokens(label) for label in labels]
    data = []
    for _, row in df.iterrows():
        question_text = row['text_train']
        positive_label_no_tokens = row['text_label']
        positive_label_with_tokens = add_special_tokens(positive_label_no_tokens)

        negative_candidates = [l for l in labels_with_tokens if l != positive_label_with_tokens]
        negative_labels = random.sample(negative_candidates, min(num_negative, len(negative_candidates)))

        data.append({
            "question": question_text,
            "positive_labels_id": labels_with_tokens.index(positive_label_with_tokens),
            "negative_labels_id": [labels_with_tokens.index(nl) for nl in negative_labels]
        })

    output = {
        "data": data,
        "labels": labels_with_tokens
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved formatted data with {len(data)} entries and {len(labels_with_tokens)} labels to {out_file}")


# Sử dụng
df = prepare_df('train.csv')
print(df.head())
create_data_and_save(df, num_negative=15)
# print
# Kiểm tra
# for i in range(min(3, len(data))):
#     print(data[i])
# print("Labels:", labels)
