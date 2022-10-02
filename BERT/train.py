from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import time

# 指標
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, cohen_kappa_score
from sklearn.model_selection import train_test_split


def f1(labels, preds): 
    return f1_score(labels, preds, average='weighted')

# 讀取訓練資料
def getDataFrame(data_path):
    df = pd.read_csv(f"{data_path}", header = None)
    df.columns = ['labels', 'text']
    
    df[['labels', 'text']] = df[['text', 'labels']]
    df.columns = ['text', 'labels']
    df["labels"] = pd.to_numeric(df["labels"])
    
    return df

def train(df):
    # 輸出語言模型的目錄名稱
    
    dir_name = 'bert-base-chinese-stock-32-epo-5'
    
    # 定義指標
    scores = {
            "acc" : accuracy_score,
            "f1" :f1,
            "kappa" : cohen_kappa_score
        }
    
    # Preparing data
    train_df, eval_df = train_test_split(df, train_size = 0.8, stratify=df['labels'])
    
    # Optional model configuration
    model_args = ClassificationArgs()
    model_args.train_batch_size = 32 # 每次放入神經網路訓練的樣本數
    model_args.eval_batch_size = 16
    model_args.num_train_epochs = 5 # 訓練回合數
    model_args.evaluate_during_training = True
    model_args.n_gpu = 2
    model_args.wandb_project = "BERT_selt_training"
    
    # 資料儲存
    model_args.output_dir = f"outputs/{dir_name}"
    model_args.best_model_dir = "outputs/best_model"
    model_args.overwrite_output_dir = True
    
    # 模型config
    model_args.max_seq_length = 512
    
    # early_stopping
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 3
    model_args.evaluate_during_training_steps = 500
    


    # Create a ClassificationModel
    model = ClassificationModel(
        'bert',
        'bert-base-chinese',
        use_cuda=True,
        num_labels=3,
        args=model_args,
    ) 
    
    # Train the model
    bert = model.train_model(train_df, eval_df = eval_df, **scores)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, **scores)
    
    return result, model_outputs, wrong_predictions

if __name__ == '__main__':
    
    data_path = "2019-04-25-done.csv"
    
    tStart = time.time()
    result, model_outputs, wrong_predictions = train(getDataFrame(data_path))
    print(result)
    print("="*30)
    print(wrong_predictions)
    tEnd = time.time() #計時結束
    print(f"執行花費 {tEnd - tStart} 秒。")   


