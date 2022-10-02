from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import time
from scipy.special import softmax
import numpy as np
# 指標
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import numpy as np
# from sklearn.semi_supervised import SelfTrainingClassifier

def f1(labels, preds): 
    return f1_score(labels, preds, average='weighted')

def predict(listTestData):
    # 輸出語言模型的目錄名稱     
    dir_name = 'best_model'
    
    # Optional model configuration
    model_args = ClassificationArgs()
    model_args.train_batch_size = 32 # 每次放入神經網路訓練的樣本數
    model_args.eval_batch_size = 16
    model_args.num_train_epochs = 5 # 訓練回合數
    model_args.evaluate_during_training = True
    model_args.n_gpu = 2
    
    # 資料儲存
    
    # 模型config
    model_args.max_seq_length = 512
    
    # early_stopping
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 3
    model_args.evaluate_during_training_steps = 500
    
    # Create a ClassificationModel
    model = ClassificationModel(
        'bert',
        f"outputs/{dir_name}",
        use_cuda=True,
        num_labels=3,
        args=model_args,
    )
    

    # Evaluate the model
    predictions, raw_outputs = model.predict(listTestData)
    
    return predictions, raw_outputs

# +
# if __name__ == '__main__':
    
#     data_path = "2022-01-30.csv"
    
#     tStart = time.time()
#     with open(f"{data_path}", 'r', encoding = 'utf-8') as f:
#         df = pd.read_csv(f, header = None)
    
#     listTestData = df[0].tolist()
#     predictions, raw_outputs = predict(listTestData)
#     print(predictions, raw_outputs)
#     tEnd = time.time() #計時結束
#     print(f"執行花費 {tEnd - tStart} 秒。")   
# -

if __name__ == '__main__':

    
    listTestData = ['.', '各位晚安，本身是剛進入股市不久的幼幼生，想知道當沖是不是需要極高技巧與抗壓性的心理特質才建議操作呢，有聽過當沖賺得錢很快，然而也聽過有人玩當沖一天賠到上百萬，是否真的是誇大了？如果只希望每年平均五到七%左右的投資報酬率，是否建議做波段就好了謝謝', '期望值每年5~7%，你乾脆買金融股存股算了，還比較穩，何必每天衝進衝出的，一不小心還會賠光本金。', '新手 我還沒見過 能穩定獲利的傢夥過新手能在股市裡撈到錢 ;只能算是運氣好(一時);好運不會永遠站在您這一方新手在股市裡賠錢;是再正常不過的事情(正常情況之下)先問問自己能有多大的屁股 可以吃多少瀉藥(輸錢)建議新手還是需要經過學習學習大約分三種1.拿真金白銀送 主力大戶 (虧錢)2.好好地 學習股市投資術 (按部就班)成長3.您自己問自己一下 ; 您準備好了嗎? (上戰場撒殺)下了多少苦工夫您只是想隨便玩玩;輸贏不計那就 隨心情而走(小賭怡情)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~買股票 很簡單 這是徒弟等級的工作賣股票 就有一定的難度了 這是師父等級的工作所有的 金融商品操作都一樣您不知道 怎麼操作的精隨 就是虧錢出場~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~先使用大腦想一下您要獲得甚麼您有下過功夫嗎?一日進入股市;終身難脫離股市天下沒有白吃的午餐要怎麼收穫 就得怎麼栽在這裡隨便 問一問 您就能賺錢呵呵呵我是不信這一套', '農曆年假期間國際油價持續攀高，每桶飆破90美元。不過，中油在加碼春節物價穩定措施下，宣布汽、柴油各吸收1.2元及2.8元，7日凌晨零時起，國內汽、柴油價格均不調整；2月14日起恢復機制調整。 因此，中油參考零售價格維持92無鉛汽油每公升29.8元、95無鉛汽油每公升31.3元、98無鉛汽油每公升33.3元、超級柴油每公升26.5元。 受大風雪襲擊美國中部頁岩油主要產地德州供應吃緊，加上烏克蘭地緣政治情勢持續緊繃，美國已決定調派兵力協防東歐與北約國家事件之助漲，導致國際油價上漲。 中油表示，本周浮動油價調整原則之調價指標7D3B週均價上漲2.31美元，新臺幣兌美元維持上週匯率，國內油價依公式計算漲幅為2.10%。 按浮動油價機制調整原則，汽、柴油每公升原應各調漲1.2元及2.8元，為維持價格低於亞洲鄰近國家(日、韓、港、星)並啟動油價平穩措施，汽、柴油每公升各共吸收0.6元及2.1元。 另為配合政府春節期間物價穩定政策及減輕民眾負擔，國內汽、柴油依「上漲不調、下跌調」的原則辦理，汽、柴油各再加碼吸收0.6元及0.7元，合計汽、柴油各共吸收1.2元及2.8元。汽、柴油價格不予調漲。 中油指出，本周依油價公式、政府調降貨物稅1元調整國內油價，並持續以亞洲國家最低價、平穩機制運作及加碼春節穩定物價措施，汽、柴油各需調整之1.2元及2.8元均由中油吸收，累計110年至12月底止，中油共吸收79.2億元。', '再漲上去，不知道加滿要加多少', '但你這代號別人看不懂 youtuber或01網友若不依照你的編碼規則發文 你看文章時豈不是又記不得?隨身攜帶 對照表?', '我只是找幾個舉例而已，沒有全部列出來……不過，被大大你猜對了，00830國泰費城半導體，我還真的取名叫“國費半”', '其實，時間久了，某些常接觸的，會自然記下來某代碼是某名稱，如 00850是元ESG 00830是國費半 0057是富摩台不常接觸的，就會用手機谷歌一下是哪一檔名稱，看到全名時，腦筋自動會轉換成簡稱', '字寫得很棒 我都寫不出方正的字體.', '過去一周，布蘭特原油期貨價格一舉突破每桶90美元，創2014年10月以來的新高，單週漲幅達5.4％，7日布油期再度站上93美元關卡，推升在台股交易的期街口布蘭特正2（00715L）今（7）日股價收盤飆漲15.78％、收在14.75元。 此外，期元大S&P石油今天收在16.33元大漲8％，成交量放大至2.6萬張，較封關日的8734張飆漲2倍。期街口布蘭特正2今天成交量也達到近5.56萬張，較封關日的2.87萬張增加近1倍。 街口投信表示，國際油價大漲，主要受助於美國德州遭逢極地寒流侵襲，溫度驟降至攝氏-10度以下，位於當地的二疊紀盆地，是美國原油的重要產區之一，因道路結冰與大量積雪，導致卡車運輸的關鍵道路受阻，加以部分地區有斷電的現象出現，進而使得美國原油供給中斷的擔憂升溫。 街口投信指出，今年異常天氣對原油供給的衝擊不僅於此，像是伊拉克國家石油行銷組織的統計顯示，伊拉克1月份的原油產量僅416萬桶/日，較前月減少63000桶/日，並低於OPEC+分配的產量目標428萬桶/日，主要是因為出口碼頭因惡劣天氣而關閉，加以碼頭儲油空間不足，間接被迫石油企業減少產量。 同一時間，奈及利亞亦是面臨同樣的困境，以致當地產量被迫減少10萬桶/日，由此可見，OPEC+難以加速增產，甚至連既定目標每月增產40萬桶/日都不易達成。街口投信表示，上述情況將有利於國際油價持續走揚。 https://www.chinatimes.com/realtimenews/20220207002422-260410', '國際油價走高，在台股封關期間布蘭特油價上漲6.99％，西德州油價上漲7.84％，帶動石油相關ETF新春開紅盤上演補漲行情，期街口布蘭特正2（00715L）、期元大S&P石油（00642U）7日盤中分別大漲16.48％及8.53％，漲勢亮麗。 群益投顧表示，影響油價波動的因素主要為烏俄衝突導致市場對於原油供應可能中斷的擔憂所致。加上OPEC增產控制得宜、美國原油產量增幅有限、庫存處於下滑的情況下，政治地緣等意外事件的發生將加深市場對於原油供需吃緊的預期，進而對油價造成較大的推升動力。 國際油價大漲，進一步拉抬石油ETF表現，期街口布蘭特正2 7日盤中大漲16.48％，暫報14.82元，成交張數逾3.4萬張；期元大S&P石油漲幅也達8.53％，股價來到16.41元，續創近兩年新高，累計今年來漲幅已超過2成。 法人指出，隨著烏俄衝突的影響已逐漸反映在油價的上漲，後續市場關注焦點應逐步回歸到基本面，包括OPEC增產進度、以及美國原油鑽井數是否出現較大增幅將為觀察重點。 https://www.chinatimes.com/realtimenews/20220207001385-260410', '謝謝分享評價方式!!可供當作另一種參考~~', '聯電is a joke', '您內行。總有人會守不住亂拋售。想想吧！有人會好心出報告叫你跑。下次再出報告叫你回來。金主買報告買新聞，引導市場。', '今天更像除權完貼息...', '建案＋商用不動產雙主軸，富旺總銷金額突破170億元...', '蝗蟲人上電視就表示股票要跌了 多年來都是如此 很少例外.', '年前買的就先放著，低點再補! (這幾天先忙事情去)', '可以到165反詐騙網站檢舉或帶證據到各地派出所報案檢舉']
    predictions, raw_outputs = predict(listTestData)
    print(predictions, raw_outputs)
    tEnd = time.time() #計時結束
    print(f"執行花費 {tEnd - tStart} 秒。")   


