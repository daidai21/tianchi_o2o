# -*- coding: utf-8 -*-


'''
using: 复现0.786的代码
    xgboost代码模型训练完成后出bug，这个是读取模型并预测
link：https://github.com/Mryangkaitong/python-Machine-learning/tree/master/Xgboost
auc：
'''


# 解决报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split



dataset1 = pd.read_csv('ProcessDataSet1.csv')
dataset1.label.replace(-1,0,inplace=True) 
dataset2 = pd.read_csv('ProcessDataSet2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('ProcessDataSet3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','coupon_id','day_gap_after'],axis=1)

dataset3.drop_duplicates(inplace=True)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

dataTrain = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataTest = xgb.DMatrix(dataset3_x)



# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1] 
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr,tpr))
    return np.average(aucs)


dataset3.drop_duplicates(inplace=True)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]



# In 【5】：
model1=xgb.Booster()
model1.load_model('xgbmodel1') 

temp = dataset12[['coupon_id','label']].copy()
temp['pred'] =model1.predict(xgb.DMatrix(dataset12_x))
temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))
print(myauc(temp))


# In 【6】：
dataset3_preds2 = dataset3_preds
dataset3_preds2['label'] = model1.predict(dataTest)
dataset3_preds2.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds2.label.values.reshape(-1,1))
dataset3_preds2.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds2.to_csv("xgb_preds2.csv",index=None,header=None)
print(dataset3_preds2.describe())


# In 【8】：
feature_score = model1.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)  # value逆序排序

fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
 
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)
