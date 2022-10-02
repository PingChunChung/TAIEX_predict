from random import uniform
from time import sleep
import os
import json

from fake_useragent import UserAgent
import cloudscraper

from datetime import timezone, datetime
import dateutil.parser

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 送requests
def response_clScraper(url):
    ua = UserAgent(cache = True)
    my_headers = {
        'User-Agent': ua.random
        }
    
    new_body = cloudscraper.create_scraper(
                    browser={
                        'browser': 'firefox',
                        'platform': 'windows',
                        'mobile': False
                }).get(url, headers = my_headers)
    
    return new_body

def get_content_json(article_id, file_name = ""):
    file_name = file_name if file_name else article_id
    file_path = "Dcard_data"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    url = f"https://www.dcard.tw/service/api/v2/posts/{article_id}"

    Flag = 0
    while Flag == 0:
        print(f"article id:{article_id} is scraping")
        new_body = response_clScraper(url)
        if new_body.status_code == 200:
            obj = new_body.json()
            
            # 調整時間格式
            time = obj['createdAt']
            dateObject = dateutil.parser.isoparse(time)
            localdt = dateObject.replace(tzinfo = timezone.utc).astimezone(tz=None)
            article_post_time = localdt.strftime("%Y-%m-%d %H:%M")
            
            # 需要的資料
            article_title = obj['title']
            article_content = obj['content']
            article_commentCount = obj['commentCount']
            article_totalCommentCount = obj['totalCommentCount']
            
            # 把資料存成dict
            data_dict = {
                "id":article_id,
                "createdAt":article_post_time,
                "title":article_title,
                "url":url,
                "content":article_content,
                "commentCount" : article_commentCount,
                "totalCommentCount": article_totalCommentCount
                }
            # 如果檔案不存在就直接寫入，存在就先讀取加上資料再寫入
            if not os.path.exists(f'{file_path}\{file_name}.json'):
                with open(f'{file_path}\{file_name}.json', "w", encoding ='utf-8') as f:
                    json.dump([data_dict], f, ensure_ascii= False, indent = 4)
            else:
                with open(f'{file_path}\{file_name}.json', "r", encoding ='utf-8') as f:
                    data = json.load(f)
                    data.append(data_dict)
                with open(f'{file_path}\{file_name}.json', "w", encoding ='utf-8') as f:
                    json.dump(data, f, ensure_ascii= False, indent = 4)
            # 抓到資料跳出迴圈
            Flag = 1
            sleep(uniform(120,180))
        else: # 沒抓到資料，重抓一次
            print(f"{article_id} can't connect: {new_body.status_code}")
            print(f"{article_id} is retrying")
            sleep(uniform(120,180))

            
def get_comment_json(article_id, file_name = ""):
    sleep_time = 120
    file_name = file_name if file_name else article_id
    file_path = "Dcard_data"
    
    url = f"https://www.dcard.tw/service/api/v2/posts/{article_id}/comments"
    
    while True:
        new_body = response_clScraper(url)
        if new_body.status_code == 200:
            print(f"article id:{article_id} commment is scraping")
            obj = new_body.json()
            if obj != []:
                data_list = []
                for data in obj:
                    # 抓取hidden為False的留言，因被刪除的留言hidden為true
                    if data['hidden'] == False and data['hiddenByAuthor']== False:
                        # 調整時間格式
                        time = data['createdAt']
                        dateObject = dateutil.parser.isoparse(time)
                        localdt = dateObject.replace(tzinfo = timezone.utc).astimezone(tz=None)
                        post_time = localdt.strftime("%Y-%m-%d %H:%M")
                        
                        comment_id = data['id']
                        content = data['content']
                        try:
                            subCommentCount = data['subCommentCount']
                        except:
                            subCommentCount = 0
                    
                        data_dict = {
                            "id":comment_id,
                            "createdAt":post_time,
                            "content":content,
                            "subCommentCount" : subCommentCount,
                            }
                        
                    
                        sub_comment_list = []
                        if subCommentCount > 0:
                            url = f"https://www.dcard.tw/service/api/v2/posts/{article_id}/comments" + f"?parentId={comment_id}"
                            new_body = response_clScraper(url)
                            
                            while True:
                                if new_body.status_code == 200:
                                    print(f"{comment_id} \nsubComment is scarping\n")
                                    sub_obj = new_body.json()
                                    
                                    # 抓子留言
                                    for sub_data in sub_obj:
                                        if sub_data['hidden'] == False and sub_data['hiddenByAuthor'] == False: 
                                            sub_time = sub_data['createdAt']
                                            sub_dateObject = dateutil.parser.isoparse(sub_time)
                                            sub_localdt = sub_dateObject.replace(tzinfo = timezone.utc).astimezone(tz=None)
                                            sub_post_time = sub_localdt.strftime("%Y-%m-%d %H:%M")
                                            
                                            
                                            sub_content = sub_data['content']
                                            
                                            sub_comment_list.append({
                                                "createdAt":sub_post_time,
                                                "content":sub_content
                                                    })
                                    
                                    data_dict.update({"subComment":sub_comment_list})
                                    sleep(uniform(sleep_time,sleep_time+60))
                                    break
                                print(f"{comment_id} can't connect: {new_body.status_code}")
                                print(f"{comment_id} is retrying\n")
                                sleep(uniform(sleep_time,sleep_time+60))
                            
                        data_list.append(data_dict)
                if not os.path.exists(f"{file_path}\{file_name}.json"):
                    with open(f"{file_path}\{file_name}.json", 'w', encoding="utf-8") as file:
                        json.dump(data_list, file, ensure_ascii=False, indent = 4)
                else:
                    with open(f"{file_path}\{file_name}.json", 'r', encoding="utf-8") as file:
                        content_json = json.load(file)
                        for content in  content_json:
                            if str(article_id) == content['id']:
                                content.update({"comment":data_list})
                    with open(f'{file_path}\{file_name}.json', "w", encoding ='utf-8') as file:
                        json.dump(content_json, file, ensure_ascii= False, indent = 4)
                break
            else:
                sleep(uniform(sleep_time,sleep_time+60))
                break
        else:
            sleep(uniform(sleep_time,sleep_time+60))
    
        
    

if __name__ == "__main__":    
    file_num = 141
    
    end = file_num

        
    path_list = sorted(os.listdir(r".\Dcard_article_id")[file_num-1: end])
    for file_name in path_list:
        data_length = 0
        if os.path.exists(f"Dcard_data\{file_name.split('.')[0]}.json"):
            with open(f"Dcard_data\{file_name.split('.')[0]}.json", 'r', encoding = "utf-8") as json_file:
                data = json.load(json_file)
                data = data[0:-1]
                data_length = len(data)
            with open(f"Dcard_data\{file_name.split('.')[0]}.json", 'w', encoding = "utf-8") as f:
                json.dump(data, f, ensure_ascii= False, indent = 4)
        file = open(f"Dcard_article_id\{file_name}", 'r', encoding = "utf-8")
        print(f"{file_name} start")
        total_id = len(file.readlines())
        file.seek(0,0)
        for index,line in enumerate(file.readlines()[data_length:]):
            article_id = line.strip("\n")
            # 檔案內id完成的進度
            # 檔案完成的進度
            print(f"current status : {data_length+index+1}/{total_id}")
            get_content_json(article_id, file_name.split('.')[0])
            get_comment_json(article_id, file_name.split('.')[0])
            print(f'completed : {path_list.index(file_name)}/{len(path_list)}' + '\n')
        print('\n' + f"{file_name} has done. ")
        print("="*40 + '\n')
    print("All done, you can close the program.")
