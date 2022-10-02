import requests as req
from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
import time
import random
import os
import re

# 抓一頁所有的url
def get_one_page(url):
    ua = UserAgent(cache = True)
    
    my_headers = {
        'user-agent': ua.random
        }
    rep = req.get(url, headers = my_headers)
    
    if rep.status_code == 200:
        soup = bs(rep.text, 'lxml')
        titles_url_temp = soup.select('div.title > a')
        titles_url = [ i['href'] for i in titles_url_temp ]
        
        file_path = 'ptt_article_url'
        regex = r'\d+'
        prefix = 'https://www.ptt.cc'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = re.search(regex, url)
        with open(f'{file_path}\{file_name[0]}.txt', 'w', encoding="utf-8") as f:
            for line in titles_url:
                f.write(prefix + line + '\n')

# 為了減少爬蟲時間，直接從ptt找到日期的index頁
def get_many_pages(a, b):
    start = a
    end = b
    
    for i in range(start, end-1, -1):
        url = "https://www.ptt.cc/bbs/Stock/index" + str(i) + ".html"
        
        get_one_page(url)
        
        time.sleep(random.random() + 1)
        
        with open("run.txt", 'a', encoding="utf-8") as temp:
            temp.write(f"{str(i)} done" + "\n")
            
# get_many_pages(5368, 40)
get_one_page("https://www.ptt.cc/bbs/Stock/index2271.html")