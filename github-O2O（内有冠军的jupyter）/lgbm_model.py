import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from time import time

import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, auc
from function import Tree_feature_select
from function import RFECV_feature_sel

import warnings

warnings.filterwarnings('ignore')

input_path = '../input/'
submi_path = '../submision/'




def clf_evaluate(df_Coupon_y, y_pred):
    y_pred = y_pred[:, 1]  # 因为sklearn输出每个类别的概率，手动选择1类

    df_val_auc = df_Coupon_y[['Coupon_id', 'label']]
    df_val_auc['pred_prob'] = y_pred

    # 计算平均AUC
    vg = df_val_auc.groupby(['Coupon_id'])
    aucs = []
    for i in vg:  # 这个“i”是分好组子集
        df_tem = i[1]

        if len(df_tem['label'].unique()) != 2:
            continue

        fpr, tpr, thresholds = roc_curve(df_tem['label'], df_tem['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print('平均auc为：', np.average(aucs))


if __name__ == '__main__':
    '''
    part1：数据划分、特征选择
    '''
    df_train1 = pickle.load(open(input_path + 'df_train1.pkl', 'rb'))
    df_train2 = pickle.load(open(input_path + 'df_train2.pkl', 'rb'))
    df_test = pickle.load(open(input_path + 'df_test.pkl', 'rb'))

    X_train1 = df_train1.drop(['label', 'Date_received', 'Date', 'diff', 'User_id', 'Coupon_id', 'Merchant_id'], axis=1)
    y_train1 = df_train1['label']

    X_train2 = df_train2.drop(['label', 'Date_received', 'Date', 'diff', 'User_id', 'Coupon_id', 'Merchant_id'], axis=1)
    y_train2 = df_train2['label']

    X_test = df_test.drop(['Date_received', 'User_id', 'Coupon_id', 'Merchant_id'], axis=1)

    # seltor, X_train2, X_train1, X_test = RFECV_feature_sel(X_train2, y_train2, X_train1, X_test)
    # feat_import, X_train2, X_train1, X_test = Tree_feature_select(X_train2, y_train2, X_train1, y_train1, X_test, 80)

    print("X_train2.shape:", X_train2.shape)
    print("X_train1.shape:", X_train1.shape)
    print("X_test.shape:", X_test.shape)


    '''
    part2：训练预测
    '''
    start_time = time()

    clf = lgb.LGBMClassifier(n_estimators=10000,
                             learning_rate=0.06,
                             max_depth=5,
                             num_leaves=30,
                             objective='binary',
                             subsample=0.9,
                             sub_feature=0.9,
                             )
    clf.fit(X_train2, y_train2, eval_set=[(X_train1, y_train1)],
            eval_metric='binary_logloss', early_stopping_rounds=100, verbose=100, )
    y_pred = clf.predict_proba(X_train1, num_iteration=clf.best_iteration)
    feat_impo = sorted(zip(X_train2.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    clf_evaluate(df_train1, y_pred)

    print('训练预测的时间为:', int(time() - start_time))


    '''
    part3：线上训练与预测，生成提交
    '''
    ###数据集合并
    X_train1 = pd.DataFrame(X_train1)
    X_train2 = pd.DataFrame(X_train2)
    X_train_all = pd.concat([X_train1, X_train2]).reset_index(drop=True)
    y_train_all = pd.concat([y_train1, y_train2]).reset_index(drop=True)

    ###合并的数据集上训练和预测
    clf = lgb.LGBMClassifier(n_estimators=180,
                             learning_rate=0.06,
                             max_depth=5,
                             num_leaves=30,
                             objective='binary',
                             subsample=0.9,
                             sub_feature=0.9,
                             )
    clf.fit(X_train_all, y_train_all)
    y_pred = clf.predict_proba(X_test, num_iteration=180)
    y_pred = y_pred[:, 1]

    ###生成提交
    submission = pd.read_csv(input_path + 'ccf_offline_stage1_test_revised.csv')[
        ['User_id', 'Coupon_id', 'Date_received']]
    df_prob = pd.DataFrame(y_pred, columns=['Probability'])
    submission = pd.concat([submission, df_prob], axis=1)
    submission.to_csv(submi_path + '8_12_all_数据集合并！！！.csv', index=False, header=None)

    print('game over')
