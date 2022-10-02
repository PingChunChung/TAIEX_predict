#!/usr/bin/env python
# coding: utf-8

# In[2]:

import yfinance as yf
#下載台灣加權指數數據(股票代號or代碼，起始日期，結束日期)
df = yf.download("^TWII", start="2017-01-01", end = "2022-09-01")
df


# In[4]:

#計算漲跌數值，ex:2017-01-04的收盤價減掉2017-01-03的收盤價，以此類推
close = df['Close'].shift(-1)-df['Close']
#漲跌值取道小數第二位
up_or_down = round(close,2)
#將漲跌值新增至一個column
df["up_or_down"]=up_or_down
df
# In[ ]:


test= df["up_or_down"].shift(1)
test[0] = 19.38
test[-1] = 141.81
df["points"] =test
df


# In[ ]:

#判斷漲跌，漲=1，跌=0
total =[]
for i in df["points"]:
    if i > 0:
        total.append(1)
    else:
        total.append(0)

#print(total)
df["status"] = total
del df["up_or_down"]
df


# In[ ]:

#另存成csv檔
df.to_csv("TAIEX_new.csv")


# In[ ]:

#繪圖
import matplotlib.pyplot as plt
df.plot(y='points')

