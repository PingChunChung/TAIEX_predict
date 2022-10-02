from random import choice, uniform
from time import sleep
import os

from fake_useragent import UserAgent
import cloudscraper



def get_article_id(forum, pages, articles=100, last_id=""):
    sucess = 0
    fail = 0
    page = 0
    page_attr = f'&before={last_id}' if last_id else ""
    while page < pages:
        page += 1
        # proxy池
        # with open(r"..\..\proxy\proxy_list.txt", 'r', encoding="utf-8") as file:
        #     proxy_lixt = [ proxy for proxy in  file.readlines()]
        # proxy = choice(proxy_lixt)
        # my_proxies = {
        #     'http': "http://" + proxy,
        #     'https': "https://" + proxy
        #     }
        
        # user=agent
        ua = UserAgent(cache = True)
        my_headers = {
            'User-Agent': ua.random
            }
        
        url = f"https://www.dcard.tw/service/api/v2/forums/{forum}/posts?popular=false&limit={articles}" + page_attr
        new_body = cloudscraper.create_scraper().get(url, headers = my_headers)
        
        # 連網有成功就把id存起來
        if new_body.status_code ==200:
            obj = new_body.json()
        
            arr = [obj[i]['id'] for i in range(len(obj))]
            
            file_path = 'Dcard_article_id'
            file_name = arr[-1]
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(f'{file_path}\{file_name}.txt', 'w', encoding="utf-8") as f:
                for line in arr:
                    f.write(str(line) + '\n')
            page_attr = f'&before={arr[-1]}'
            sucess += 1
            print(f"id : {arr[-1]}")
            print(f"sucess :{sucess}")
            print(f"fail :{fail}")
            print(f"total :{sucess+fail}" + "\n")
            sleep(uniform(120,180))
        # 失敗就讓迴圈次數多一次         
        else:
            page -= 1
            fail += 1
            print(f"sucess :{sucess}")
            print(f"fail :{fail}")
            print(f"total :{sucess+fail}" + "\n")
            sleep(uniform(120,180))
            
get_article_id("stock", 60, last_id=235665464)