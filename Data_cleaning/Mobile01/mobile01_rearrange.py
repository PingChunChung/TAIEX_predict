import json, os, re

# 放json檔的資料夾
file = 'mobile01_data'
allFile = os.listdir(file)
# allFile = ['mobile01_real_data.json']
def rearrange(target_time, target_content):
    time_len = len(target_time)
    if time_len == 15 or time_len == 16:
        csv_name =target_time[0:10]  
        

        # 寫入留言內容
        content = re.sub(r',+|\n+', ' ', target_content)
        content = re.sub(r'^\s+|\s+$', '', content)
        if content != "" :
            f = open(f"mobile01_csv/{csv_name}_mo.csv",'a',encoding = 'utf-8')
            f.write(content+"\n")
            f.seek(0,0)
            f.close()
    else:
        print(file_name,'日期格式有例外')
    

if __name__ == "__main__":

    # 刪除mobile01_csv/所有檔案
    csv_file = 'mobile01_csv'
    for f in os.listdir(csv_file):
        os.remove(f'{csv_file}/{f}')
    
    for file_name in allFile:
        with open(f'mobile01_data/{file_name}','r',encoding = 'utf-8') as json_obj:
            data = json.load(json_obj)
            for article in range(len(data)):          
                rearrange(data[article]['createdAt'], data[article]['content'])
                
                try:
                    comment = data[article]['comment']
                    if comment != [] :
                        for usr_com in range(len(comment)):
                            rearrange(comment[usr_com]['comment_time'], comment[usr_com]['comment_content'])
                        
                except:
                    continue
                
                try:
                    subComment = data[article]['subComment']
                    if subComment != [] :
                        for usr_com in range(len(subComment)):
                            rearrange(subComment[usr_com]['subComment_time'], subComment[usr_com]['subComment_content'])
                            
                except:
                    continue
    print('Done!')
