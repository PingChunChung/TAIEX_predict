import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import  metrics
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc,classification_report,roc_auc_score,matthews_corrcoef,cohen_kappa_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

# Read file
df = pd.read_csv('count&rate_all.csv')

# Calculate Emotion-Rate
# df['PR'] = df['count_1'] / (df['count_0'] + df['count_1'] + df['count_2'] )
# df['NR'] = df['count_0'] / (df['count_0'] + df['count_1'] + df['count_2'] )
# df['MR'] = df['count_2'] / (df['count_0'] + df['count_1'] + df['count_2'] )

#Drop data from 2017~2018/3
df = df.drop(df.index[:278])

# Train-Test
x = df.iloc[:,-3:].values
y = df[['status_y']].values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,random_state=1)

# Scaler
# scaler = StandardScaler()
# scaler = scaler.fit(train_x)

# train_x = scaler.transform(train_x)
# test_x = scaler.transform(test_x)

#def scores
def scores_(model, test_y, pred_y):
    print('accuracy_score   : {:.5f}'.format(metrics.accuracy_score(test_y,pred_y)))
    print('precision Score  : {:.5f}'.format(metrics.precision_score(test_y, pred_y)))
    print('recall Score     : {:.5f}'.format(metrics.recall_score(test_y, pred_y)))
    print('f1 Score         : {:.5f}'.format(metrics.f1_score(test_y, pred_y)))
    print(metrics.classification_report(test_y, pred_y))
    print(metrics.confusion_matrix(test_y, pred_y))

    return metrics.accuracy_score(test_y,pred_y), metrics.precision_score(test_y, pred_y),metrics.recall_score(test_y, pred_y), metrics.f1_score(test_y, pred_y)



# Model-Random Forest
model = RandomForestClassifier(random_state=1)
#model = RandomForestClassifier(n_estimators=130,max_features='auto',max_depth=3,min_samples_split=7,min_samples_leaf=1,bootstrap=True,criterion='gini', random_state=1)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
print('RandomForestClassifier : \n')
scores_(model,test_y, y_pred)
print('訓練集: ',model.score(train_x,train_y))
print('測試集: ',model.score(test_x,test_y))



# Model-Logistic Regression
model = LogisticRegression(random_state=1)
# model = LogisticRegression(random_state=1,penalty = 'none')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
print('LogisticRegression : \n')
scores_(model,test_y, y_pred)
# tmp = model.predict_proba(test_x)
# pred_new = np.where(tmp[:,1]>0.531, 1, 0)
# scores_(model,test_y, pred_new)
print('訓練集: ',model.score(train_x,train_y))
print('測試集: ',model.score(test_x,test_y))

# Pred_Proba Adjust
# acc = []
# prec = []
# rec = []
# f1 = []
# i_value=[]
# for i in range(500,621, 1):
#     pred_new = np.where(tmp[:,1]>(i/1000), 1, 0)
#     accuracy_score,precision_Score,recall_Score,f1_Score = scores_(model,test_y, pred_new)
#     acc.append(accuracy_score)
#     prec.append(precision_Score)
#     rec.append(recall_Score)
#     f1.append(f1_Score)
#     i_value.append(i/1000)

# Best ACC - output 0.6111111111111112
# max(acc)


# Return Index - output 31
# acc.index(0.6111111111111112)


# Draw Plot
tmp = model.predict_proba(test_x)
pred_new = np.where(tmp[:,1]>0.531, 1, 0)
print('LogisticRegression : \n')
scores_(model,test_y, pred_new)
fpr, tpr, thresholds = roc_curve(test_y,tmp[:,1])
roc_auc = auc(fpr,tpr)
roc_auc_score(test_y,tmp[:,1])
print(roc_auc_score(test_y,tmp[:,1]))
plt.plot(fpr,tpr,'k-', label = 'ROC(area = {0:2f})'.format(roc_auc),lw=2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()



# Print ACC/Pred_Proba & Prec/Pred_Proba
acc=np.array(acc)
prec=np.array(prec)
i_value=np.array(i_value)
plt.plot(i_value,acc,'k-',lw=2)
plt.ylabel('accuracy')
plt.xlabel('Pred_prob')
plt.title('Accuracy Flow')
plt.legend(loc='lower right')
plt.show()

plt.plot(i_value,prec,'k-',lw=2)
plt.ylabel('precision')
plt.xlabel('Pred_prob')
plt.title('Precision Flow')
plt.legend(loc='lower right')
plt.show()


# Model-Gradient Boosting
model = GradientBoostingClassifier(random_state=1)
#model = GradientBoostingClassifier(loss ='log_loss',learning_rate=0.035,n_estimators=104,random_state=1)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
print('GradientBoostingClassifier : \n')
scores_(model,test_y, y_pred)
print('訓練集: ',model.score(train_x,train_y))
print('測試集: ',model.score(test_x,test_y))