import pandas as pd
from datasets import Dataset
from random import randint,sample

class DataProcess :
    def __init__(self,train_data_path,new_train_data_path):
        self.train_data_path = train_data_path
        self.new_train_data_path = new_train_data_path
        self.labels = []
        self.misconceptions = []

    def format_train_data(self):
        df = pd.read_csv(self.train_data_path,keep_default_na=False)

        df['label'] = 1
        df['row_id'] = df['row_id'].astype('int64')
        df['QuestionId'] = df['QuestionId'].astype('int64')
        df['QuestionText'] = df['QuestionText'].astype('str')
        df['MC_Answer'] = df['MC_Answer'].astype('str')
        df['StudentExplanation'] = df['StudentExplanation'].astype('str')
        df['Category'] = df['Category'].astype('str')
        df['Misconception'] = df['Misconception'].astype('str')
        df['label'] = df['label'].astype('int64')
        df.to_csv(self.train_data_path,index=False)
        misconceptions = []
        for i in range(len(df)):
            if misconceptions.count(df.iloc[i]['Misconception'])==0 and ('Misconception' in df.iloc[i]['Category']) :
                misconceptions.append(df.iloc[i]['Misconception'])


        labels = ["True_Neither","False_Neither","True_Correct","False_Correct"]

        for i in misconceptions :
            candidate1 = str("True_Misconception"+":"+i)
            candidate2=str("False_Misconception"+":"+i)
            labels.append(candidate1)
            labels.append(candidate2)
        # print(label)

        self.labels = labels

        return df,labels
    
    def create_negative_train_data(self,df,labels):
        negative_data = []
        size = len(df)
        count = 0
        true = 1
        false = -1
        NA_label = ["True_Neither","False_Neither","True_Correct","False_Correct"]
        for index,row in df.iterrows():
            negative_sample = dict(row)
            negative_data.append(negative_sample)
        for index, row in df.iterrows():
            negative_labels = sample(range(len(labels)),7)
            for i in negative_labels:
                label = labels[i]
                category = None
                misconception = None
                if label in NA_label :
                    category = label
                    misconception ="NA"
                else:
                    parts = label.split(":")
                    category = parts[0]
                    misconception=parts[1]
                negative_sample = {
                    'row_id': row['row_id'] + size + count,
                    'QuestionId': row['QuestionId'],
                    'QuestionText': row['QuestionText'],
                    'MC_Answer': row['MC_Answer'],
                    'StudentExplanation': row['StudentExplanation'],
                    'Category': category,
                    'Misconception': misconception,
                    'label': false
                }
                count+=1
                negative_data.append(negative_sample)
        
        
        negative_df = pd.DataFrame(negative_data)
        
        
        negative_df['row_id'] = negative_df['row_id'].astype('int64')
        negative_df['QuestionId'] = negative_df['QuestionId'].astype('int64')
        negative_df['QuestionText'] = negative_df['QuestionText'].astype('str')
        negative_df['MC_Answer'] = negative_df['MC_Answer'].astype('str')
        negative_df['StudentExplanation'] = negative_df['StudentExplanation'].astype('str')
        negative_df['Category'] = negative_df['Category'].astype('str')
        negative_df['Misconception'] = negative_df['Misconception'].astype('str')
        negative_df['label'] = negative_df['label'].astype('int64')
        negative_df.to_csv(self.new_train_data_path, index=False)
        print(f"New dataset saved to: {self.new_train_data_path}")
        
        return negative_df


# if __name__ == "__main__":
#     TRAIN_PATH = "train_fixed.csv"
#     NEG_PATH ="training_dataset.csv"
#     data_process= DataProcess(TRAIN_PATH,NEG_PATH)
#     df,labels = data_process.format_train_data()
#     data_process.create_negative_train_data(df,labels)