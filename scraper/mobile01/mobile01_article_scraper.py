from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from time import sleep
from datetime import datetime
import os
import json



ua = UserAgent(cache = True)
my_options = webdriver.ChromeOptions()
# my_options.add_argument("--headless")  # 背景執行
my_options.add_argument("--incognito")  # 無痕模式
my_options.add_argument("user-agent={}".format(ua.random)) # 更改ua
my_options.add_argument("--disable-blink-features=AutomationControlled") # 跳過防爬蟲
# my_options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" #使用Brave瀏覽器


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
sleep(5)
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
print(f"Hello {account}, you've loged in mobile01")
sleep(5)


page_prefix = "https://www.mobile01.com/topiclist.php?f=793&p="

driver.get(f'{page_prefix}1')
soup = bs(driver.page_source, 'lxml')

# 找出最後一頁的頁數
pages = soup.find_all("a", class_ = "c-pagination")
page_list = [ i['data-page'] for i in pages]
final_page = max(int(p) for p in page_list)


article_urls = []
article_titles = []
article_post_time = []
data_list = []

# 進入股票版的每一頁
for p in range(1,final_page+1):
    
    driver.get(f'{page_prefix}{p}')

    soup = bs(driver.page_source, 'lxml')
    article_url_temp = soup.find_all("a", class_="c-link u-ellipsis")
    article_post_time_temp = soup.select("div.o-fNotes")

    # 每篇文章的連結、標題及發文時間
    article_urls.extend( [i['href'] for i in article_url_temp])
    article_titles.extend([i.text for i in article_url_temp])
    article_post_time.extend([i.text for index,i in enumerate(article_post_time_temp) if index%2 == 0])
    
    print(f'Page {p}/{final_page} is done')
    sleep(5)


now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
data_list.append({
    "time":now,
    "total_pages":final_page,
    "total_articles":len(article_titles)
})

for i in range(len(article_titles)):
    data_list.append({
        "createdAt":article_post_time[i],
        "title":article_titles[i],
        "url":prefix+article_urls[i],
    })

with open('./mobile01-articles', 'w', encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii = False, indent=4)

print('All is done')

driver.quit() # 結束後執行


