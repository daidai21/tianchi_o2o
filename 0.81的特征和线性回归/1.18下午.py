# -*- coding: utf-8 -*-


'''
using：使用100行代码入门天池O2O优惠券使用新人赛【精简教程版】的模型和0.81的特征集
auc：
'''


# import package
import pandas as pd


# path
path = 'cache_天池o2o参考代码.py_train.csv'


# train
def train():
    pass


# prediction
def prediction():
    pass


# main call
if __name__ == "__main__":
    # 读取数据 训练输几局
    data = pd.read_csv(path)

    X = data.drop(columns='Coupon_id')
    print(X.info())
    print(X.head(5))

    '''
    预测使用的样本集 不知道怎么进行特征提取
    '''
