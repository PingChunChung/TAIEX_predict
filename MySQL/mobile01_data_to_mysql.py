import json, pymysql, os, re


def read_json():
    file_path = f'C:/Share/Final Project/{source}/{source}_data'
    files = os.listdir(file_path)

    data = []
    for file in files:
        with open(f'{file_path}/{file}','r',encoding='utf-8') as f:
            lines  = f.read()
            data.extend(json.loads(lines)) # 解析每一行資料
    return data
    
def get_articles(data):
    articles = []
    for article in data:
        # 取得文章url編號作為文章id
        url = article["url"]
        f = re.search( r'(?<=f=)\d+', url).group()
        t = re.search( r'(?<=t=)\d+', url).group()
        article_id = f + t
        
        articles.append((article_id,
                         source,
                         article['createdAt'],
                         article['url'],
                         article['title'],
                         article['content']))
    print("articles get")
    return articles

def get_comments(data):
    comments = []
    for article in data:
        # 取得文章url編號作為文章id
        url = article["url"]
        f = re.search( r'(?<=f=)\d+', url).group()
        t = re.search( r'(?<=t=)\d+', url).group()
        article_id = f + t
        
        if article['comment'] != [] :
            comment_list = article['comment']
            for comment in comment_list:
                comments.append((article_id,
                                 "comment",
                                 comment['comment_time'],
                                 comment['comment_content']
                                 ))
                                
        if article['subComment']  != [] :
            subComment_list = article['subComment']
            for subComment in subComment_list:
                comments.append((article_id,
                                 "subComment",
                                 subComment['subComment_time'],
                                 subComment['subComment_content']
                                 ))
    print("comments get")
    return comments

def prem(db):
    cursor = db.cursor()
    cursor.execute("SELECT VERSION()")
    data = cursor.fetchone()
    print("Database version : %s " % data) # 結果表明已經連線成功
    cursor.execute("DROP TABLE IF EXISTS comments") # 習慣性
    cursor.execute("DROP TABLE IF EXISTS articles") # 習慣性
    create_table_articles = """CREATE TABLE articles (
                                                        id VARCHAR(20) NOT NULL PRIMARY KEY,
                                                        source VARCHAR(10),
                                                        created_time VARCHAR(100),
                                                        url VARCHAR(100),
                                                        title TEXT,
                                                        content MEDIUMTEXT
                                                        )"""
    cursor.execute(create_table_articles)  
    print("TABLE articles is created")
    
    create_table_comments = """CREATE TABLE comments (
                                                        id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
                                                        article_id VARCHAR(20),
                                                        level VARCHAR(10),
                                                        created_time VARCHAR(100),
                                                        content MEDIUMTEXT,
                                                        FOREIGN KEY(article_id) REFERENCES articles(id)
                                                        )"""
    cursor.execute(create_table_comments) # 根據需要建立一個表格    
    print("TABLE comments is created")
    
def artcles_insert(db, artcles):
    check = set()
    for i, article in enumerate(artcles):
        print(f'正在載入第{i}筆文章......')
        if article not in check:
            try:
                result = [article]
                inesrt_re = "INSERT INTO articles(id, source, created_time, url, title, content) values (%s,%s,%s,%s,%s,%s)"
                cursor = db.cursor()
                cursor.executemany(inesrt_re,result)
                db.commit()
              
            except Exception as e:
                db.rollback()
                print(str(e))
                break
        check.add(article)

def comments_insert(db, comments):
    check = set()
    for i, comment in enumerate(comments):
        print(f'正在載入第{i}筆留言......')
        if comment not in check:
            try:
                result = [comment]
                inesrt_re = "INSERT INTO comments(article_id, level, created_time, content) values (%s,%s,%s,%s)"
                cursor = db.cursor()
                cursor.executemany(inesrt_re,result)
                db.commit()
              
            except Exception as e:
                db.rollback()
                print(str(e))
                break
        check.add(comment)


source = "mobile01"
data = read_json()
artcles = get_articles(data)
comments = get_comments(data)
db = pymysql.connect( host = "localhost", user = "SqlManager", password = "Passw0rd!", database = "team3",charset='utf8mb4')
cursor = db.cursor()
prem(db)
artcles_insert(db, artcles)
comments_insert(db, comments)
cursor.close()
