# -*- coding: utf-8 -*-


'''
using: 复现0.786的代码
link：https://github.com/Mryangkaitong/python-Machine-learning/tree/master/Xgboost
auc：
'''


# 解决报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# In 【1】：
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


# In 【3】：
params={'booster':'gbtree',
        'objective': 'rank:pairwise',
        'eval_metric':'auc',
        'gamma':0.1,
        'min_child_weight':1.1,
        'max_depth':5,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'colsample_bylevel':0.7,
        'eta': 0.01,
        'tree_method':'exact',
        'seed':0,
        'nthread':12
        }
watchlist = [(dataTrain,'train')]
model = xgb.train(params,dataTrain,num_boost_round=3500,evals=watchlist)

model.save_model('xgbmodel')
model=xgb.Booster(params)
model.load_model('xgbmodel') 
# predict test set 
dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = model.predict(dataTest)
dataset3_preds1.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds1.label.values.reshape(-1,1))

dataset3_preds1.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds1.to_csv("xgb_preds.csv",index=None,header=None)
print(dataset3_preds1.describe())


# In 【3】：
model=xgb.Booster()
model.load_model('xgbmodel') 

temp = dataset12[['coupon_id','label']].copy()
temp['pred'] = model.predict(xgb.DMatrix(dataset12_x))
temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))
print(myauc(temp))


# In 【4】：
params={'booster':'gbtree',
        'objective': 'rank:pairwise',
        'eval_metric':'auc',
        'gamma':0.1,
        'min_child_weight':1.1,
        'max_depth':5,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'colsample_bylevel':0.7,
        'eta': 0.01,
        'tree_method':'exact',
        'seed':0,
        'nthread':12
        }

cvresult = xgb.cv(params, dataTrain, num_boost_round=20000, nfold=5, metrics='auc', seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(50)
        ])
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

watchlist = [(dataTrain,'train')]
model1 = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)

model1.save_model('xgbmodel1')
print('------------------------train done------------------------------')


# In 【5】：
model1 = xgb.Booster()
model1.load_model('xgbmodel1') 

temp = dataset12[['coupon_id','label']].copy()
temp['pred'] = model1.predict(xgb.DMatrix(dataset12_x))
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



'''
0.921424


0.8067332521264581
复现预测.py:68: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  dataset3_preds2['label'] = model1.predict(dataTest)
/Users/daidai/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py:4405: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self[name] = value
复现预测.py:70: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  dataset3_preds2.sort_values(by=['coupon_id','label'],inplace=True)
            user_id      coupon_id  date_received          label
count  1.128030e+05  112803.000000   1.128030e+05  112803.000000
mean   3.684618e+06    9064.658006   2.016072e+07       0.395830
std    2.126358e+06    4147.283515   9.017693e+00       0.122418
min    2.090000e+02       3.000000   2.016070e+07       0.000000
25%    1.843824e+06    5035.000000   2.016071e+07       0.322636
50%    3.683073e+06    9983.000000   2.016072e+07       0.380615
75%    5.525176e+06   13602.000000   2.016072e+07       0.458327
max    7.361024e+06   14045.000000   2.016073e+07       1.000000
'''
