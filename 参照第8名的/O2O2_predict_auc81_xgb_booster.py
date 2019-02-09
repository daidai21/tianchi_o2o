# 电脑的bug而添加的
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve


def get_processed_data():
    dataset1 = pd.read_csv('data_preprocessed_2/ProcessDataSet1.csv')
    dataset2 = pd.read_csv('data_preprocessed_2/ProcessDataSet2.csv')
    dataset3 = pd.read_csv('data_preprocessed_2/ProcessDataSet3.csv')

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset12.fillna(0, inplace=True)
    dataset3.fillna(0, inplace=True)

    return dataset12, dataset3


def train_xgb(dataset12, dataset3):
    predict_dataset = dataset3[['User_id', 'Coupon_id', 'Date_received']].copy()
    predict_dataset.Date_received = pd.to_datetime(predict_dataset.Date_received, format='%Y-%m-%d')
    predict_dataset.Date_received = predict_dataset.Date_received.dt.strftime('%Y%m%d')

    # 将数据转化为dmatric格式
    dataset12_x = dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id', 'label'], axis=1)
    dataset3_x = dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], axis=1)

    train_dmatrix = xgb.DMatrix(dataset12_x, label=dataset12.label)
    predict_dmatrix = xgb.DMatrix(dataset3_x)

    # xgboost模型训练
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
            #   'tree_method': 'gpu_hist',
            #   'n_gpus': '-1',
              'seed': 0,
              'nthread': cpu_jobs,
            #   'predictor': 'gpu_predictor'
              }

    # 使用xgb.cv优化num_boost_round参数
    cvresult = xgb.cv(params, train_dmatrix, num_boost_round=10000, nfold=2, metrics='auc', seed=0, callbacks=[
        xgb.callback.print_evaluation(show_stdv=False),
        xgb.callback.early_stop(50)
    ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)

    # 使用优化后的num_boost_round参数训练模型
    watchlist = [(train_dmatrix, 'train')]
    model = xgb.train(params, train_dmatrix, num_boost_round=num_round_best, evals=watchlist)

    model.save_model('train_dir_2/xgbmodel')
    params['predictor'] = 'cpu_predictor'
    model = xgb.Booster(params)
    model.load_model('train_dir_2/xgbmodel')

    # predict test set
    dataset3_predict = predict_dataset.copy()
    dataset3_predict['label'] = model.predict(predict_dmatrix)

    # 标签归一化
    dataset3_predict.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        dataset3_predict.label.values.reshape(-1, 1))
    dataset3_predict.sort_values(by=['Coupon_id', 'label'], inplace=True)
    dataset3_predict.to_csv("train_dir_2/xgb_preds.csv", index=None, header=None)
    print(dataset3_predict.describe())

    # 在dataset12上计算auc
    # model = xgb.Booster()
    # model.load_model('train_dir_2/xgbmodel')

    temp = dataset12[['Coupon_id', 'label']].copy()
    temp['pred'] = model.predict(xgb.DMatrix(dataset12_x))
    temp.pred = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
    print(myauc(temp))


# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['Coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    dataset12, dataset3 = get_processed_data()
    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_xgb(dataset12, dataset3)

    # print('predict: start predicting......')
    # # predict('xgb')
    # print('predict: predicting finished.')

    # log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    # log += '----------------------------------------------------\n'
    # open('%s.log' % os.path.basename(__file__), 'a').write(log)
    # print(log)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %s s' % (datetime.datetime.now() - start).seconds)


'''
num_boost_round:6699


[17:31:29] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 62 extra nodes, 0 pruned nodes, max_depth=5
[6697]	train-auc:0.922968
[17:31:30] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 56 extra nodes, 0 pruned nodes, max_depth=5
[6698]	train-auc:0.92297
            User_id      Coupon_id          label
count  1.128030e+05  112803.000000  112803.000000
mean   3.684618e+06    9064.658006       0.090630
std    2.126358e+06    4147.283515       0.166669
min    2.090000e+02       3.000000       0.000000
25%    1.843824e+06    5035.000000       0.010282
50%    3.683073e+06    9983.000000       0.029689
75%    5.525176e+06   13602.000000       0.070591
max    7.361024e+06   14045.000000       1.000000
0.8116708416604529
2019-02-09 17:32:35
time costed is: 4327 s
'''
