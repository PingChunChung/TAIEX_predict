import json
import os
import re

# ptt清整

# 放json檔的資料夾
file = 'ptt_data'
allFile = os.listdir(file)


# (實作)讀json檔的文章內容轉為file_name
for file_name in allFile:
    with open(f'ptt_data/{file_name}','r',encoding = 'utf-8') as json_obj:
        data = json.load(json_obj)
        for article in range(len(data)):
            
            # 寫入文章內容
            title = data[article]['標題'][:4]
            article_time = data[article]['時間']
            article_detail = data[article]['文章內容']
            article_comment = data[article]['留言']
            
            # 刪除網頁連結 & 簽名檔的正規表達式
            cls_sign = r'(--(?<=--).*?(?=--)--)'
            cls_http = r'http\S+'
            cls_www = r'www\S+'
            cls_png = r'\S*.png'
            cls_jpg = r'\S*.jpg'
            # cls_blank = r'\s\.?'
            
            # 判斷是否是公告 or 創作 or 問卷
            if title != '[公告]' or  title != '[創作]' or title != '[問卷]':
                if (len(article_time)) == 16:
                    
                    csv_name = article_time[0:len(article_time)-6]
                    f = open(f"ptt_csv(date)/{csv_name}.csv",'a',encoding = 'utf-8')
                    
                    # 寫入文章內容
                    # 去簽名檔
                    
                    # 若無簽名檔
                    mark = re.findall(cls_sign,article_detail)[::-1]
                    if mark == []:
                        
                        # 去網頁
                        tmp1 = re.sub(cls_http,'',article_detail)
                        tmp2 = re.sub(cls_www,'',tmp1)
                        tmp3 = re.sub(cls_png,'',tmp2)
                        tmp4 = re.sub(cls_jpg,'',tmp3)
    
                        # 去空格
                        # tmp3 = re.sub(cls_blank,'',tmp2)
    
                        # \n換空格
                        article_detail2 = tmp4.replace('\n',' ')
                        
                        # 半形','改成空格
                        article_detail3 = article_detail2.replace(',',' ')
    
                        if article_detail3 != '':
                            # 寫入
                            # f.write(file_name)
                            f.write(article_detail3)
                        
                    # 有簽名檔
                    else:
                        
                        # 移除簽名檔
                        tmp = article_detail.replace(re.findall(cls_sign,article_detail)[-1],'')
    
                        # 去網頁
                        tmp1 = re.sub(cls_http,'',tmp)
                        tmp2 = re.sub(cls_www,'',tmp1)
                        tmp3 = re.sub(cls_png,'',tmp2)
                        tmp4 = re.sub(cls_jpg,'',tmp3)
    
                        # 去空格
                        # tmp3 = re.sub(cls_blank,'',tmp2)
    
                        # \n換空格
                        article_detail2 = tmp4.replace('\n',' ')
                        
                        # 半形','改成空格
                        article_detail3 = article_detail2.replace(',',' ')
    
                        if article_detail3 != '':
                            # 寫入
                            # f.write(file_name)
                            f.write(article_detail3)
                    
                                                    
                # 文章個位數日期加 0 
                elif (len(article_time)) == 15:
                    top = article_time.index('-')+4
                    bot = article_time.index(' ')
                    csv_name = (article_time[:top]) + '0' + (article_time[top:bot])
                    f = open(f"ptt_csv(date)/{csv_name}.csv",'a',encoding = 'utf-8')
                    
                    # 寫入文章內容
                    
                    # 去簽名檔        
                    # 若無簽名檔
                    mark = re.findall(cls_sign,article_detail)[::-1]
                    if mark == []:
                        # 去網頁
                        tmp1 = re.sub(cls_http,'',article_detail)
                        tmp2 = re.sub(cls_www,'',tmp1)
                        tmp3 = re.sub(cls_png,'',tmp2)
                        tmp4 = re.sub(cls_jpg,'',tmp3)
    
                        # 去空格
                        # tmp3 = re.sub(cls_blank,'',tmp2)
    
                        # \n換空格
                        article_detail2 = tmp4.replace('\n',' ')
                        
                        # 半形','改成空格
                        article_detail3 = article_detail2.replace(',',' ')
    
                        if article_detail3 != '':
                            # 寫入
                            # f.write(file_name)
                            f.write(article_detail3)
                        
                    # 有簽名檔
                    else:
                        
                        # 移除簽名檔
                        tmp = article_detail.replace(re.findall(cls_sign,article_detail)[-1],'')
    
                        # 去網頁
                        tmp1 = re.sub(cls_http,'',tmp)
                        tmp2 = re.sub(cls_www,'',tmp1)
                        tmp3 = re.sub(cls_png,'',tmp2)
                        tmp4 = re.sub(cls_jpg,'',tmp3)
    
                        # 去空格
                        # tmp3 = re.sub(cls_blank,'',tmp2)
    
                        # \n換空格
                        article_detail2 = tmp4.replace('\n',' ')
                        
                        # 半形','改成空格
                        article_detail3 = article_detail2.replace(',',' ')
    
                        if article_detail3 != '':
                            # 寫入
                            # f.write(file_name)
                            f.write(article_detail3)
                    
                        
                else:
                    continue
                
                f.write('\n')
                f.seek(0)
                
              # 寫入留言內容
                for usr_com in range(len(article_comment)):
    
                    comment_time = data[article]['留言'][usr_com]['時間']
                    comment_id = data[article]['留言'][usr_com]['ID']              
                    comment_detail = data[article]['留言'][usr_com]['內容']
                    
                    
                    if ':' in comment_time:
                        where_colon = comment_time.index(':')
                        
    
                        # 時間格式有含ip
                        if (len(comment_time)) > 16 and usr_com < (len(article_comment) -1):
                            
                            next_comment_id  = data[article]['留言'][usr_com +1]['ID']
    
                            # 日期設為檔名
                            year = comment_time[:5]
                            date = comment_time[where_colon - 8:where_colon - 3]
                            detail_name = year + date
                            
                            # 和下一個不同id
                            if (comment_id != next_comment_id):
                                
                                f_usr_com =  open(f"ptt_csv(date)/{detail_name}.csv",'a',encoding = 'utf-8')
                                # 寫入留言內容
                                # 去網頁
                                tmp = re.sub(cls_http,'',comment_detail)
                                tmp2 = re.sub(cls_www,'',tmp)
                                tmp3 = re.sub(cls_png,'',tmp2)
                                tmp4 = re.sub(cls_jpg,'',tmp3)
    
                                # 去空格
                                # tmp3 = re.sub(cls_blank,'',tmp2)
                                
                                
                                # 半形','改成空格
                                comment_detail1 = tmp4.replace(',',' ')
    
                                if article_detail3 != '':
                                    
                                    # 寫入
                                    # f_usr_com.write(file_name)
                                    f_usr_com.write(comment_detail1+'\n')
                                    
                    
                            else:
                                # 和下一個同id
                                f_usr_com =  open(f"ptt_csv(date)/{detail_name}.csv",'a',encoding = 'utf-8')
    
                                # 寫入留言內容
                                # 去網頁
                                tmp = re.sub(cls_http,'',comment_detail)
                                tmp2 = re.sub(cls_www,'',tmp)
                                tmp3 = re.sub(cls_png,'',tmp2)
                                tmp4 = re.sub(cls_jpg,'',tmp3)
    
                                # 去空格
                                # tmp3 = re.sub(cls_blank,'',tmp2)
                                
                                # 半形','改成空格
                                comment_detail1 = tmp4.replace(',',' ')
    
                                if comment_detail1 != '':    
                                    # 寫入
                                    # f_usr_com.write(file_name)
                                    f_usr_com.write(comment_detail1)
                                
    
                        # 時間格式正常    
                        elif (len(comment_time)) == 16 and usr_com < (len(article_comment) -1):   
                            
                            next_comment_id  = data[article]['留言'][usr_com +1]['ID']
                            
                            # 和下一個不同id
                            if  (comment_id != next_comment_id):
                                
                                detail_name = comment_time[:10]
                                f_usr_com =  open(f"ptt_csv(date)/{detail_name}.csv",'a',encoding = 'utf-8')
                                
                                # 寫入留言內容
                                # 去網頁
                                tmp = re.sub(cls_http,'',comment_detail)
                                tmp2 = re.sub(cls_www,'',tmp)
                                tmp3 = re.sub(cls_png,'',tmp2)
                                tmp4 = re.sub(cls_jpg,'',tmp3)
    
                                # 去空格
                                # tmp3 = re.sub(cls_blank,'',tmp2)
                
                                # 半形','改成空格
                                comment_detail1 = tmp4.replace(',',' ')
    
                                if comment_detail1 != '':
                                    # 寫入
                                    # f_usr_com.write(file_name)
                                    f_usr_com.write(comment_detail1 + '\n')
                                    
                                
                            else:
                                # 和下一個同id
                                detail_name = comment_time[:10]
                                f_usr_com =  open(f"ptt_csv(date)/{detail_name}.csv",'a',encoding = 'utf-8')
                                
                                # 寫入留言內容
                                # 去網頁
                                tmp = re.sub(cls_http,'',comment_detail)
                                tmp2 = re.sub(cls_www,'',tmp)
                                tmp3 = re.sub(cls_png,'',tmp2)
                                tmp4 = re.sub(cls_jpg,'',tmp3)
    
                                # 去空格
                                # tmp3 = re.sub(cls_blank,'',tmp2)
                                
                                
                                # 半形','改成空格
                                comment_detail1 = tmp4.replace(',',' ')
    
                                if comment_detail1 != '':
                                    # 寫入
                                    # f_usr_com.write(file_name)
                                    f_usr_com.write(comment_detail1)
    
                        else:
                            detail_name = comment_time[:10]
                            f_usr_com =  open(f"ptt_csv(date)/{detail_name}.csv",'a',encoding = 'utf-8')
                            # 寫入留言內容
                            # 去網頁
                            tmp = re.sub(cls_http,'',comment_detail)
                            tmp2 = re.sub(cls_www,'',tmp)
                            tmp3 = re.sub(cls_png,'',tmp2)
                            tmp4 = re.sub(cls_jpg,'',tmp3)

                            # 去空格
                            # tmp3 = re.sub(cls_blank,'',tmp2)

                            # 半形','改成空格
                            comment_detail1 = tmp4.replace(',',' ')

                            if comment_detail1 != '':
                                # 寫入
                                # f_usr_com.write(file_name)
                                f_usr_com.write(comment_detail1 + '\n')
                            
                    else:
                        # 沒有冒號就排除
                        continue
                




# (實作)關閉 csv檔
f.close()
f_usr_com.close()
print('結束')
