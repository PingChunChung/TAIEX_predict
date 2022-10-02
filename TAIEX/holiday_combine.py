#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import pendulum


#列出所有檔名

datapath = 'BDSE_26_t3_project/data/predicted_data/Dcard_csv(data)'
files = os.listdir(datapath)

#移除副檔名

re_file = []
for file in files:
    basename = os.path.basename(file)
    file_name = os.path.splitext(basename)[0]
    re_file.append(file_name[:10])
    
# 查找星期
week = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
week_list = []
for i in re_file:
    a =pendulum.parse(i).day_of_week
    week_list.append(week[a])
    


rate0_list = []
rate1_list = []
rate2_list = []
count0_list = []
count1_list = []
count2_list = []

# 迴圈計算比例   
for file in files:
    df = pd.read_csv(f'BDSE_26_t3_project/data/predicted_data/Dcard_csv(data)/{file}')
    
    #計算當日總則數
    label_count = df['label'].value_counts()
 
    
    # 計算 0,1,2個數
    try:
        count_0=label_count[0]
    except:
        count_0 = 0
        
    try:
        count_1=label_count[1]
    except:
        count_1 = 0
        
    try:
        count_2=label_count[2]
    except:
        count_2 = 0
        
    total = count_0 + count_1 + count_2
    
    
    #寫入lsit
    count0_list.append(count_0)
    count1_list.append(count_1)
    count2_list.append(count_2)
    
    
    #計算比例 並寫入list
    rate_0 = count_0 / total
    rate0_list.append(rate_0)
    rate_1 = count_1 / total
    rate1_list.append(rate_1)
    rate_2 = count_2 / total
    rate2_list.append(rate_2)



# 將上述所有list 寫成DataFrame 寫入CSV
header = {'Date':re_file,
          'Week':week_list,
          'count_0':count0_list,
          'count_1':count1_list,
          'count_2':count2_list,
          'rate_0':rate0_list,
          'rate_1':rate1_list,
          'rate_2':rate2_list
          }
header_df = pd.DataFrame(header)
header_df.to_csv("dcard/Dcard(output)/final_dcard.csv")



# merge 漲跌幅資料
# count data (left)
df_1 = header_df[['Date','count_0','count_1','count_2']]

# status data (right)
df_2 = pd.read_csv("TAIEX_new.csv")
df_TAI = df_2[['Date','status']]

# merge
df_TAI_new = pd.merge(df_1,df_TAI, on=['Date'],how='left')

df_TAI_new.to_csv("dcard/Dcard(output)/final_merge_dcard.csv")



# 讀入merge檔
data = pd.read_csv("dcard/Dcard(output)/final_merge_dcard.csv")
X = pd.DataFrame(data)

# 加總沒有status的天數
try:
    for i in range(len(X['status'])):
        if X["status"][i] != 1 and X["status"][i] != 0:
            X["count_0"][i+1] += X["count_0"][i]
            X["count_1"][i+1] += X["count_1"][i]
            X["count_2"][i+1] += X["count_2"][i]

        else :
            continue
            
except:
    pass

A = X.copy()

# 去除空值
A_drop = A.dropna()
A_drop.to_csv("dcard/Dcard(output)/final_sumweek_dcard.csv")

