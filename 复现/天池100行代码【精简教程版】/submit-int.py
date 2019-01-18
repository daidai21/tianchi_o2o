# -*- coding: utf-8 -*-


'''
using：对天池100行代码【精简版】预测的数据进行向上取整
    link：天池100行代码【精简教程版】.py
auc：
'''


import numpy as np
import pandas as pd
import copy


data_df = pd.read_csv('submit-1.csv')
data_np = data_df.values


# 第四列乘10
data_np[:][3] = data_np[:][3] * 10
print('* 10:', data_np[:][:5])


# 向上取整
data_np_up = copy.deepcopy(data_np)
data_np_up[:][3] = np.floor(data_np_up[:][3])
print('up int:', data_np_up[:][:5])
data_np_up[:][3] = data_np_up[:][3] / 10
print('save', data_np_up[:][:5])
data_pd_up = pd.DataFrame(data_np_up)
print(data_pd_up.info())
# data_pd_up.round({'0': 0, '1': 0, '2': 0, '3': 1})
# data_pd_up.to_csv('submit-up.csv')


# 向下取整
data_np_down = copy.deepcopy(data_np)
data_np_down[:][3] = np.ceil(data_np_down[:][3])
print('down int:', data_np_down[:][:5])
data_np_down[:][3] = data_np_down[:][3] / 10
print('save:', data_np_down[:][:5])
data_pd_down = pd.DataFrame(data_np_down)
print(data_pd_down.info())
# data_pd_down.round({'0': 0, '1': 0, '2': 0, '3': 1})
# data_pd_down.to_csv('submit-down.csv')
