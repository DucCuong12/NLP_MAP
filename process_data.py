from sklearn.preprocessing import LabelEncoder
import pandas as pd
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