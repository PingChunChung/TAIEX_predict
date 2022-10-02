from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from time import sleep
import json
import csv
import re

ua = UserAgent(cache = True)
my_options = webdriver.ChromeOptions()
# my_options.add_argument("--headless")  # 背景執行
my_options.add_argument("--incognito")  # 無痕模式
my_options.add_argument("user-agent={}".format(ua.random)) # 更改ua
my_options.add_argument("--disable-blink-features=AutomationControlled") # 跳過防爬蟲
#my_options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" #使用Brave瀏覽器


# 開啟瀏覽器
driver = webdriver.Chrome(
    options = my_options,
    service = Service(ChromeDriverManager().install())
)

ua = UserAgent(cache = True)

my_headers = {
    'user-agents':ua.random
}
prefix = "https://www.mobile01.com/"

# 進入mobile01
driver.get(prefix)

# 登入
account = "yourAccount"
password = "yourPassword"

login = driver.find_element(By.CSS_SELECTOR, "div.l-header__main > div > div.l-header__tools > div > div.l-headerTools__login > a")
login.click()
fill_account = driver.find_element(By.ID,"regEmail")
fill_account.send_keys(account)
fill_password = driver.find_element(By.ID,"regPassword")
fill_password.send_keys(password)
submit = driver.find_element(By.ID,"submitBtn")
driver.execute_script("arguments[0].click();", submit)
print(f"Hello {account}, you've loged in mobile 01")
sleep(2)

# 讀取待爬清單
missed_article_urls = []
with open('mobile01_missed', 'r', encoding='UTF8') as f:
    reader = csv.reader(f)
    for row in reader:
        missed_article_urls.extend(row)

article_titles = []
article_post_time = []
article_content = []
comment_list = []
subComment_list = []
article_num = 0
for article_url in missed_article_urls:
    
    driver.get(article_url)
    soup = bs(driver.page_source, 'lxml')
    
    # 每篇文章的標題、發文時間、內文
    article_titles.append(soup.select_one("div.l-docking__title > div > div > h1").get_text())
    article_post_time.append(soup.select_one("div.l-navigation__item.is-dockingHide > ul > li:nth-child(1) > span").get_text())
    article_content.append(soup.select_one("article> div").get_text())

    comment_time = []
    comment_content = []
    comment = []
    subComment_time = []
    subComment_content = []
    subComment = []

    # 找到留言最後一頁的頁數
    comment_pages = soup.find_all("a", class_ = "c-pagination")
    comment_page_list = [ i['data-page'] for i in comment_pages]
    if comment_page_list != [] : 
        final_comment_page = max(int(cp) for cp in comment_page_list)
    else:
        final_comment_page = 1
        

    for f in range(1,final_comment_page+1):
        
        driver.get(f'{article_url}&p={f}')
        soup = bs(driver.page_source, 'lxml')
        # 文章留言
        comment_time_temp = soup.select("div.l-articlePage__publish > div.l-navigation > div:nth-child(1) > span:nth-child(1)")
        comment_time.extend([ i.text for i in comment_time_temp])

        comment_content_temp = soup.select("div.u-gapBottom--max.c-articleLimit>article")
        comment_content.extend([ i.text for i in comment_content_temp])
        comment_content = [ re.sub(r"\s*.* wrote:.*\s", "", c) for c in comment_content]
        
        subComment_time_temp = soup.select(" div.msgTool.l-navigation.u-gapNextV > div:nth-child(1) > span")
        subComment_time.extend([ i.text for i in subComment_time_temp])

        subComment_content_temp = soup.select("div > div.msgContent.c-summary__desc")
        subComment_content.extend([ i.text for i in subComment_content_temp])
            
        sleep(2)
        
    for c in range(len(comment_content)):
        comment.append({"comment_time" : comment_time[c], "comment_content" : comment_content[c]})
    for s in range(len(subComment_content)):    
        subComment.append({"subComment_time" : subComment_time[s], "subComment_content" : subComment_content[s]})
    comment_list.append(comment)
    subComment_list.append(subComment)
    
    article_num += 1
    print(f'article {article_num}/{len(missed_article_urls)} is done')


data_list = []
    
for i in range(len(missed_article_urls)):
    data_list.append({
            "createdAt":article_post_time[i],
            "title":article_titles[i],
            "url":missed_article_urls[i],
            "content":article_content[i],
            "comment":comment_list[i],
            "subComment":subComment_list[i]
    })
with open('mobile01_data/mobile01-updated_data', 'w', encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii = False, indent=4)
print('All is done')
    
driver.quit() # 結束後執行

