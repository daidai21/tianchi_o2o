# -*- coding:utf-8 -*-


'''
using:融合gdbt和xgb模型（gdbt*0.7 + xgb*0.3）
notice:融合前要确保本身的submi.csv文件中的用户id、优惠券id、时间对应
'''


# import lib
import copy
import numpy as np
import pandas as pd


# settings path
path_gdbt = 'gdbt.csv'
path_xgb = 'xgb.csv'


# read csv
gdbt_df = pd.read_csv(path_gdbt, names=['user_id', 'coupon_id', 'date', 'rate'])
xgb_df = pd.read_csv(path_xgb, names=['user_id', 'coupon_id', 'date', 'rate'])


# see data type
# print('===== gdbt =====')
# print(gdbt_df.head())
# print(gdbt_df.dtypes)
# print(gdbt_df.describe())
# print('===== xgb =====')
# print(xgb_df.head())
# print(xgb_df.dtypes)
# print(xgb_df.describe())


# df to np
gdbt_np = np.array(gdbt_df)
xgb_np = np.array(xgb_df)
del gdbt_df
del xgb_df
# print('gdbt_np', gdbt_np)
# print('xgb_np', xgb_np)


# print len(row) and len(col)
# print(gdbt_np.shape)  # (112803, 4)
# print(xgb_np.shape)  # (112803, 4)


# xgb * 0.3 + gdbt * 0.7
submit_np = copy.deepcopy(gdbt_np)  # deep copy
for row in range(gdbt_np.shape[0]):  # fuse
    submit_np[row][3] = 0.3 * gdbt_np[row][3] + 0.7 * xgb_np[row][3]
# print('submit_np', submit_np)
# submit_df = pd.DataFrame(submit_np, columns=['user_id', 'coupon_id', 'date', 'rate'], dtype=['int64', 'int64', 'int64', 'float64'])
submit_df = pd.DataFrame(submit_np, columns=['user_id', 'coupon_id', 'date', 'rate'])
del submit_np  # save memary


# submit set data type int
submit_df['user_id'] = submit_df['user_id'].astype('int')
submit_df['coupon_id'] = submit_df['coupon_id'].astype('int')
submit_df['date'] = submit_df['date'].astype('int')


# print submit df information
print(submit_df)
print(submit_df.head())
print(submit_df.dtypes)
print(submit_df.describe())


# save submit to csv
submit_df.to_csv('submit-(0.3gdbt-0.7xgb).csv', header=False, index=False)
