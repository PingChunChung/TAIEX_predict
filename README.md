# TAIEX_predict
使用BERT的預訓練模型，將網路文章及留言做文字情緒分類，區分
出正面、負面及中立情緒，並利用機器學習預測股票大盤漲跌，以此
分析網路輿情是否影響股市，並將結果以網頁呈現給使用者。

## scraper
爬取Dcard、Mobile01、PTT三大論壇的資料

PTT使用一般request爬蟲  
```
ptt_article_url.py # 獲取文章的url
get_content_json.py # 從文章url爬取所有文章的資訊
```

Dcard透過api2.0進行爬蟲
```

Dcard_article_url.py  # 獲得Dcard文章的編號
get_content_json.py # 透過文章編號，分別使用API的文章內容、留言、以及子留言獲取內容
```

Mobile01使用selenium
```
mobile01_article_scraper.py : 爬取mobile01股票版的文章連結
check_scrapy.py : 比對現有資料，確認需要補爬的文章連結(第一次爬可以跳過這個步驟，但是要先去改3.的檔案路徑)
mobile01_data_scraper.py : 爬取文章內容
```

## TAIEX
獲取大盤加權指數

使用yfinance套件獲取大盤加權指數


## MySQL
使用PyMySQL將原始資料匯入資料庫

## Data_cleaning
將不必要之資料，例如標點符號、網址連結、格式不正確之資料等整理成需要格式，  
並轉成csv檔。

## BERT
第一階段模型  
協助判斷文章正負向情緒

## Predict_model
將BERT所得出的結果作為Feature，  
以此預測隔日開盤大盤加權指數的漲跌

## Web
將結果以網頁呈現
