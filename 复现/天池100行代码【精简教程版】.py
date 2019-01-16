# -*- coding: utf-8 -*-
'''
using：复现（100行代码入门天池O2O优惠券使用新人赛【精简教程版】）
link：https://tianchi.aliyun.com/notebook-ai/detail?postId=8462
auc：0.52054151
'''

'''
# import package
import pickle
import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression


# 读取数据
dfoff = pd.read_csv('dataset/ccf_offline_stage1_train.csv')
dftest = pd.read_csv('dataset/ccf_offline_stage1_test_revised.csv')
dfon = pd.read_csv('dataset/ccf_online_stage1_train.csv')
print('data read end.')


# 1. 将满xx减yy类型(`xx:yy`)的券变成折扣率 : `1 - yy/xx`，同时建立折扣券相关的特征 `discount_rate, discount_man, discount_jian, discount_type`
# 2. 将距离 `str` 转为 `int`
# convert Discount_rate and Distance
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def getDiscountMan(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


print("tool is ok.")
# tool is ok.


#
def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df


dfoff = processData(dfoff)
dftest = processData(dftest)

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[pd.notnull(date_received)])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date'])
couponbydate = dfoff[dfoff['Date_received'].notnull()][[
    'Date_received', 'Date'
]].groupby(
    ['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received', 'count']
buybydate = dfoff[(dfoff['Date'].notnull())
                  & (dfoff['Date_received'].notnull())][[
                      'Date_received', 'Date'
                  ]].groupby(
                      ['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received', 'count']

print("end")

# [ 1.          0.86666667  0.95        0.9         0.83333333  0.8         0.5
#   0.85        0.75        0.66666667  0.93333333  0.7         0.6
#   0.96666667  0.98        0.99        0.975       0.33333333  0.2         0.4       ]
# [ 0.83333333  0.9         0.96666667  0.8         0.95        0.75        0.98
#   0.5         0.86666667  0.6         0.66666667  0.7         0.85
#   0.33333333  0.94        0.93333333  0.975       0.99      ]
# end


#
def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(
    lambda x: 1 if x in [6, 7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(
    lambda x: 1 if x in [6, 7] else 0)

# change weekday to one-hot encoding
weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(
            row['Date'], format='%Y%m%d') - pd.to_datetime(
                row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0


dfoff['label'] = dfoff.apply(label, axis=1)

print("end")
# end

# data split
print("-----data split------")
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516)
           & (df['Date_received'] <= 20160615)].copy()
print("end")
# -----data split------

# feature
original_feature = [
    'discount_rate', 'discount_type', 'discount_man', 'discount_jian',
    'distance', 'weekday', 'weekday_type'
] + weekdaycols
print("----train-----")
model = SGDClassifier(  # lambda:
    loss='log',
    penalty='elasticnet',
    fit_intercept=True,
    max_iter=100,
    shuffle=True,
    alpha=0.01,
    l1_ratio=0.01,
    n_jobs=1,
    class_weight=None)
model.fit(train[original_feature], train['label'])
# ----train-----

# #### 预测以及结果评价
print(model.score(valid[original_feature], valid['label']))
# 0.909452622077


# 
print("---save model---")
with open('1_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('1_model.pkl', 'rb') as f:
    model = pickle.load(f)
# ---save model---


# test prediction for submission
y_test_pred = model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['label'] = y_test_pred[:, 1]
dftest1.to_csv('sample_submission.csv', index=False, header=False)
dftest1.head()



# myself add
print(dftest1.head())
'''

# ===== ===== ===== =====

import time
import pickle
import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression


# 计算时间
start_run = time.time()


# 读取数据
dfoff = pd.read_csv('dataset/ccf_offline_stage1_train.csv')
dftest = pd.read_csv('dataset/ccf_offline_stage1_test_revised.csv')
dfon = pd.read_csv('dataset/ccf_online_stage1_train.csv')
print('data read end.')


# 1. 将满xx减yy类型(`xx:yy`)的券变成折扣率 : `1 - yy/xx`，同时建立折扣券相关的特征 `discount_rate, discount_man, discount_jian, discount_type`
# 2. 将距离 `str` 转为 `int`
# convert Discount_rate and Distance
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def getDiscountMan(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


print("tool is ok.")
# tool is ok.


#
def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df


dfoff = processData(dfoff)
dftest = processData(dftest)

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[pd.notnull(date_received)])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date'])
couponbydate = dfoff[dfoff['Date_received'].notnull()][[
    'Date_received', 'Date'
]].groupby(
    ['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received', 'count']
buybydate = dfoff[(dfoff['Date'].notnull())
                  & (dfoff['Date_received'].notnull())][[
                      'Date_received', 'Date'
                  ]].groupby(
                      ['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received', 'count']

print("end")

# [ 1.          0.86666667  0.95        0.9         0.83333333  0.8         0.5
#   0.85        0.75        0.66666667  0.93333333  0.7         0.6
#   0.96666667  0.98        0.99        0.975       0.33333333  0.2         0.4       ]
# [ 0.83333333  0.9         0.96666667  0.8         0.95        0.75        0.98
#   0.5         0.86666667  0.6         0.66666667  0.7         0.85
#   0.33333333  0.94        0.93333333  0.975       0.99      ]
# end


#
def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(
    lambda x: 1 if x in [6, 7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(
    lambda x: 1 if x in [6, 7] else 0)

# change weekday to one-hot encoding
weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(
            row['Date'], format='%Y%m%d') - pd.to_datetime(
                row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0


dfoff['label'] = dfoff.apply(label, axis=1)

print("end")
# end

# data split
print("-----data split------")
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516)
           & (df['Date_received'] <= 20160615)].copy()
print("end")
# -----data split------

# feature
original_feature = [
    'discount_rate', 'discount_type', 'discount_man', 'discount_jian',
    'distance', 'weekday', 'weekday_type'
] + weekdaycols


# 读取模型
with open('1_model.pkl', 'rb') as f:
    model = pickle.load(f)
# 预测
y_test_pred = model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['label'] = y_test_pred[:, 1]


# 计算时间
end_run = time.time()
print('计算时间： \t', end_run - start_run)

# 查看
print(dftest1.head())
dftest1 = dftest1.round({'User_id': 0, 'Coupon_id': 0, 'Date_received': 0, 'label': 1})
print(dftest1.head())

# 保存
dftest1.to_csv('sample_submission.csv', index=False, header=False)


'''
预测的全是0.0, 0.1, 0.2
'''
