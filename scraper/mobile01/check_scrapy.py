'''
確認目前有哪些mobile01的文章是還沒被爬下來的，並存成一個list的csv
'''

import json
import os
import csv

file = 'mobile01_data'
allFile = os.listdir(file)

all_urls = set()
scraped_urls = set()

# 打開mobile01文章總表
with open('mobile01-articles', 'r', encoding="utf-8") as f:
    articles = json.load(f)
    # 把文章url存成set
    for a in range(2,len(articles)):
        all_urls.add(articles[a]["url"])

# 打開目前爬到的所有資料
for file_name in allFile:
    with open(f'{file}/{file_name}', 'r', encoding="utf-8") as f:
        data = json.load(f)
        # 把文章url存成set
        for d in range(len(data)):
            scraped_urls.add(data[d]["url"])

# 相減得到少爬的文章url，存成list
missed_articles = list(all_urls - scraped_urls)

print("all articles:" , len(all_urls))
print("scraped content:" , len(scraped_urls))
print("need to be scraped:" , len(missed_articles))

# 存成csv
with open('mobile01_missed', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(missed_articles)
