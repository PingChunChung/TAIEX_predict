import requests as req
from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
import json
import os

from random import random
import time

from ptt_article_url import get_one_page

# def get_file_names(start, end):
#     file_names = []
#     for i in range(start, end + 1, 1):
#         file_names.append(os.popen(f"dir /b .\ptt_article_url\{i}.txt").read().strip("\n"))
#     return file_names

# file_names = get_file_names(60, 62)

# start, end 為檔名從哪裡開始
def get_json(start, end):
    month = {
        'Jan' : '01',
        'Feb' : '02',
        'Mar' : '03',
        'Apr' : '04',
        'May' : '05',
        'Jun' : '06',
        'Jul' : '07',
        'Aug' : '08',
        'Sep' : '09', 
        'Oct' : '10',
        'Nov' : '11',
        'Dec' : '12'
        }
    # 創資料存放的目錄
    file_path = "ptt_data"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    for i in range(start, end + 1):
        try:  # 如果沒有網址的txt檔，就製作一份
            file = open(f"ptt_article_url\{i}.txt", 'r', encoding = "utf-8")
        except FileNotFoundError:
            get_one_page(f"https://www.ptt.cc/bbs/Stock/index{i}.html")
            file = open(f"ptt_article_url\{i}.txt", 'r', encoding = "utf-8")
        
        finally:
            data_list = []    
            for line in file.readlines():
                url = line.strip("\n")
                
                ua = UserAgent(cache = True)
                my_headers = {
                    'user-agent': ua.random
                    }
                res = req.get(url, headers = my_headers)
                
                if res.status_code == 200 :
                    print(f"index {i}  :scraping")
                    soup = bs(res.text, 'lxml')
                    
                    # 避開被刪文的文章
                    if soup.title.string == "閱讀文章 - 看板 Stock - 批踢踢實業坊":
                        continue
                    
                    # 抓表頭的資訊
                    article_header = soup.select("span.article-meta-value")
                    try:
                        article_title = article_header[-2].text
                    except IndexError:
                        article_title = article_header[0].text
                        article_post_time = ""
                        
                    article_post_time_temp = article_header[-1].text.split()
                    
                    # 如果沒有時間會報錯，報錯後修正表頭的資訊
                    try:
                        year = article_post_time_temp[4] # 給下面comment用
                        article_post_time = f"{article_post_time_temp[4]}-{month[article_post_time_temp[1]]}-{article_post_time_temp[2]} {article_post_time_temp[3]}"[:-3]
                    except IndexError:
                        year = ""
                        article_post_time = ""
                        article_title = article_header[-1].text
                    except KeyError:
                        continue
                    
                    
                    # 抓內文
                    article_content = soup.select_one("div#main-content").text.split('※ 發信站: 批踢踢實業坊(ptt.cc)')[0]
                    article_content =  article_content.split("\n")[1:]
                    article_content = [ i.strip() for i in article_content if i.strip() != ""]
                    article_content =  "\n".join(article_content)
                    
                    
                    
                    # 抓留言
                    comment_list = []
                    comments = soup.select("div.push")
                    
                    try:
                        for comment in comments:
                            comment_userid = comment.select_one("span.push-userid").text
                            comment_time = comment.select_one("span.push-ipdatetime").text.strip()
                            comment_content = comment.select_one("span.push-content").text.strip(": ")
                            
                            comment = {
                                "ID":comment_userid,
                                "時間":f"{year}-{comment_time.replace('/', '-')}",
                                "內容":comment_content
                                }
                            comment_list.append(comment)
                    except AttributeError:
                        pass
                    
                        
                    # 把所有資料丟進list
                    data_list.append({
                        "時間":article_post_time,
                        "標題":article_title,
                        "網址":url,
                        "文章內容":article_content,
                        "留言":comment_list
                        })
                # 寫入json
                time.sleep(random())
            with open(f'{i}.json', "w", encoding ='utf-8') as f:
                json.dump(data_list, f, ensure_ascii= False, indent = 4)
            print(f"index {i} is done")
                
                # 當個有禮貌的孩子

if __name__ == "__main__":
    get_json(5367, 5368)
                    