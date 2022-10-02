import train
import predict
from scipy.special import softmax
import numpy as np
import pandas as pd
import time


def sample_labeled_data(df):
    df.columns = ['labels', 'text']
    data0 = df[df['labels']==0]
    data1 = df[df['labels']==1]
    data2 = df[df['labels']==2]
    
    sample_n = min([len(data0),len(data1),len(data2)])
    
    df = pd.concat([data0.sample(n=sample_n),data1.sample(n=sample_n),data2.sample(n=sample_n)], ignore_index = True)
    
    return df


def self_training():
    label_data_path = "labeled_data.csv"
    unlabeled_data_path = "unlabeled_data.csv"
    predict_num = 25000
    probabilities_basic = 0.99
    # 已經label的資料

    labeled_df_origin = pd.read_csv(f"{label_data_path}", header = None)
    labeled_df = sample_labeled_data(labeled_df_origin)
    labeled_df.columns = ['labels', 'text']
    labeled_df[['labels', 'text']] = labeled_df[['text', 'labels']]
    labeled_df.columns = ['text', 'labels']
    
    tStart = time.time()
    result, model_outputs, wrong_predictions = train.train(labeled_df)

    # 還沒label的資料
    unlabeled_df = pd.read_csv(f"{unlabeled_data_path}", header = None)
    sample_unlabeled_df = unlabeled_df.sample(n=predict_num)
    unlabeled_df = unlabeled_df.drop(labels=sample_unlabeled_df.index) # 把原始資料被抽出來的刪除
    unlabeled_df.columns = ['text']
    sample_unlabeled_df = sample_unlabeled_df.reset_index(drop=True)
    
    listTestData = sample_unlabeled_df[0].tolist()
    predictions, raw_outputs = predict.predict(listTestData)
    
    
    # 把raw outputs轉成機率
    probabilities = softmax(raw_outputs, axis=1)
    probabilities_mask = probabilities>probabilities_basic # 遮罩
    high_probabilities_list = np.sum(probabilities_mask, axis=1) # 找出機率高的
    sample_unlabeled_df = pd.concat([sample_unlabeled_df, pd.Series(predictions), pd.Series(high_probabilities_list), pd.Series(list(raw_outputs))],axis=1)
    sample_unlabeled_df.columns = ['text', 'labels', 'high_probabilities','raw_outputs']
    
    # 取出要得label資料
    high_probabilities_data = sample_unlabeled_df.query('high_probabilities==1')[['text', 'labels']] # 把高機率的資料取出來
    sample_unlabeled_df = sample_unlabeled_df.drop(labels=high_probabilities_data.index).reset_index(drop=True)
    high_probabilities_data.reset_index(drop= True)
    labeled_df = pd.concat([labeled_df_origin, high_probabilities_data],ignore_index= True ) # 把labeled資料合併
    labeled_df.columns = ['labels', 'text']
    labeled_df.to_csv(f"{label_data_path}", header = False, index = False) # 把新增過的labeled資料存檔
    
    
    # 處理不符合機率的資料
    sample_unlabeled_df = sample_unlabeled_df.query('high_probabilities==0')[['text']]
    unlabeled_df = pd.concat([unlabeled_df,sample_unlabeled_df], ignore_index= True)
    unlabeled_df.to_csv(f"{unlabeled_data_path}", header = False, index = False)
    

    tEnd = time.time() #計時結束
    print(f"執行花費 {tEnd - tStart} 秒。")   

if __name__ == '__main__':
    label_data_path = "labeled_data.csv"
    labeled_df_origin = pd.read_csv(f"{label_data_path}", header = None)
    while len(labeled_df_origin) < 200000:
        self_training()
        labeled_df_origin = pd.read_csv(f"{label_data_path}", header = None)
    print("資料大於200000啦!!!!!!!")
