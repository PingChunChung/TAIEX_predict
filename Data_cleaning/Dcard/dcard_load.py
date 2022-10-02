#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import re


# In[2]:


cls_http = r'http\S+'   #擷取http
cls_www  = r'www\S+'    #擷取www
cls_emojiword = r':.*:' #擷取轉成文字的emoji
cls_Bnum = r'[bB]\d\d?-?\d?\d?\s?' #擷取(B2...)
cls_Fnum = r'\n\d\.'    #擷取開頭數字(1.)
cls_letter =r'\n[a-z].' #擷取開頭英文(a.)
cls_blank = r'\s\.?-?'    #擷取空白符號


# In[3]:


file_path = "Dcard_csv(data)"
if not os.path.exists(file_path):
        os.makedirs(file_path)

        
file = "Dcard_data"
allfile = os.listdir(file)


id_set = set()

for file_name in allfile:
    with open (f'{file}/{file_name}' , 'r' ,encoding = 'utf-8') as json_obj:
        text = json.load(json_obj)
        for obj in text:
            if obj['id'] not in id_set:
                art_date = obj["createdAt"][:10]  
                f = open (f"Dcard_csv(data)/{art_date}.csv" , "a" , encoding = 'utf-8')
                
                #去除網址連結
                no_url = re.sub(cls_http,'\n' , obj['content'])
                no_wwww = re.sub(cls_www,'\n' , no_url)
                #print(no_wwww)

                #去除開頭的數字or英文
                no_num = re.sub(cls_Fnum,'' , no_wwww)
                no_letter = re.sub(cls_letter,'' , no_num)
                #print(no_letter)


                #擷取空白符號(\r\n、\n\)
                no_rn = re.sub(cls_blank,'',no_letter)+'\n'
                #print(no_rn)
                id_set.add(obj['id'])


                f.write(no_rn)
                f.seek(0.0)
                f.close()

             
                 #判斷文章是否有留言
                if "comment" in obj:
                    for objc in obj["comment"]:
                        com_date = objc["createdAt"][:10]
                        f_comment = open (f"Dcard_csv(data)/{com_date}.csv" , "a" , encoding = 'utf-8')

                        #去除網址連結
                        comment = re.sub(cls_http,'\n',objc['content'])
                        de_www = re.sub(cls_www,'\n',comment)
                        #print(de_www)

                        #去除開頭的數字
                        cls_num = re.sub(cls_Fnum,'' , de_www)
                        #print(cls_num)

                        #去除開頭的英文
                        #clr_letter = re.sub(cls_letter,'' , cls_num)
                        #print(cls_letter)

                        #去除bB+數字
                        de_Bnum = re.sub(cls_Bnum, '\n', cls_num)
                        #print(de_Bnum)

                        #擷取空白符號(\r\n、\n\)
                        cls_rn = re.sub(cls_blank,'',de_Bnum)

                        if cls_rn != '' :
                            final_content = cls_rn+'\n'
                            #print(final_content)

                            f_comment.write(final_content)
                            f_comment.seek(0.0)
                            f_comment.close()




                            #判斷是否有流言下面是否有留言
                            if "subComment" in objc:
                                for sub_objc in objc["subComment"]:
                                    sub_com_date = sub_objc["createdAt"][:10]
                                    f_sub_comment = open (f"Dcard_csv(data)/{sub_com_date}.csv" , "a" , encoding = 'utf-8')

                                    #去除網址連結
                                    sub_comment = re.sub(cls_http,'\n',sub_objc['content'])
                                    sub_www = re.sub(cls_www,'\n',sub_comment)
                                    #print(sub_comment)

                                    #去除開頭的數字or英文
                                    clr_num = re.sub(cls_Fnum, '\n' , sub_www)
                                    #print(clr_num)

                                    #去除bB+數字
                                    clr_bnum = re.sub(cls_Bnum, '\n', clr_num)
                                    #print(clr_bnum)

                                    #將/r/n取代成/n
                                    clr_rn = re.sub(cls_blank,'',clr_bnum)

                                    if clr_rn != '' :
                                        final_sub_content = clr_rn +'\n'

                                        f_sub_comment.write(final_sub_content)
                                        f_sub_comment.seek(0.0)
                                        f_sub_comment.close()
