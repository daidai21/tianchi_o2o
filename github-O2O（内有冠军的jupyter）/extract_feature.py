###库#####################################################################
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from time import time
import random
import re
from numpy.random import randn

import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  #'all'|'last'|'last_expr'|'none'

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

input_path = '../input/'
submi_path = '../submision/'


###基础函数
def time_convert(int_time):
    if int_time != -1:
        str_time = str(int(int_time))
        year = int(str_time[0: 4])
        month = int(str_time[4: 6])
        day = int(str_time[6: 8])
        datatime_time = datetime(year, month, day)
    else:
        datatime_time = -1

    return datatime_time
def discount_rate_trans(str_rate):
    if str_rate == 'fixed':
        float_rate = 0.9
    else:
        list_ = re.findall('\d*', str_rate)
        a = int(list_[0])
        b = int(list_[2])

        if a != 0:
            float_rate = (a - b) / a
        else:
            float_rate = float(str_rate)

    return float_rate
def discount_rate_man(str_rate):
    if re.search(':', str_rate):
        list_ = re.findall('\d*', str_rate)
        a = int(list_[0])
        b = int(list_[2])

        return a
    else:
        return 0
def discount_rate_jian(str_rate):
    if re.search(':', str_rate):
        list_ = re.findall('\d*', str_rate)
        a = int(list_[0])
        b = int(list_[2])

        return b
    else:
        return 0
def time_convert(int_time):
    if int_time != -1:
        str_time = str(int(int_time))
        year = int(str_time[0: 4])
        month = int(str_time[4: 6])
        day = int(str_time[6: 8])
        datatime_time = datetime(year, month, day)
    else:
        datatime_time = -1

    return datatime_time

###制作统计特征
def feat_nunique(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].nunique().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_mode(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].mode().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_count(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].count().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_sum(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].sum().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_mean(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].mean().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_max(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].max().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_min(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].min().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar

###六大特征群
def user_off_feat(df_train1, df_train1_feat_offline):
    #用户领取优惠券的次数：
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Date_received != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['user_received_coupon_count'])

    #用户领取优惠券，但没有消费的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date == -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['user_received_coupon_butNotConsume_count'])

    #用户领取优惠券，而且消费了的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['user_received_coupon_andConsume_count'])

    #用户领取优惠券后的核销率：
    df_train1['user_received_coupon_ConsumeRate'] = df_train1.user_received_coupon_andConsume_count / df_train1.user_received_coupon_count

    #用户核销优惠券的平均折扣率，最大折扣率，最小折扣率 ：
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_tem_feat['Discount_rate'] = df_tem_feat.Discount_rate.apply(discount_rate_trans)
    df_train1 = feat_mean(df_train1, df_tem_feat, ['User_id'], ['Discount_rate'], ['user_consume_coupon_aveDiscountRate'], na=-1)
    df_train1 = feat_max(df_train1, df_tem_feat, ['User_id'], ['Discount_rate'], ['user_consume_coupon_maxDiscountRate'], na=-1)
    df_train1 = feat_min(df_train1, df_tem_feat, ['User_id'], ['Discount_rate'], ['user_consume_coupon_minDiscountRate'], na=-1)

    #用户核销过优惠券的不同商家数量；以及占所有商家的比重：
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_nunique(df_train1, df_tem_feat, ['User_id'], ['Merchant_id'], ['user_consume_coupon_MerchantNunique'])
    df_train1['user_consume_coupon_MerchantRate'] = df_train1['user_consume_coupon_MerchantNunique'] / df_tem_feat.Merchant_id.nunique()

    #用户核销过的不同优惠券数量；以及占所有核销优惠券的比重：
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_nunique(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['user_consume_coupon_Nunique'])
    df_train1['user_consume_coupon_Rate'] = df_train1['user_consume_coupon_Nunique'] / df_train1['user_received_coupon_andConsume_count']

    #用户平均核销每个商家多少张优惠券：
    df_train1['user_consume_coupon_AveCount'] = df_train1['user_received_coupon_andConsume_count'] / df_train1['user_consume_coupon_MerchantNunique']

    #用户核销优惠券中，离商家的平均距离，最远距离，最近距离
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)& 
                                         (df_train1_feat_offline.Distance != -1)].copy()
    df_train1 = feat_mean(df_train1, df_tem_feat, ['User_id'], ['Distance'], ['user_consume_coupon_meanDistance'], na=-1)
    df_train1 = feat_max(df_train1, df_tem_feat, ['User_id'], ['Distance'], ['user_consume_coupon_maxDistance'], na=-1)
    df_train1 = feat_min(df_train1, df_tem_feat, ['User_id'], ['Distance'], ['user_consume_coupon_minDistance'], na=-1)

    return df_train1
def user_on_feat(df_train1, df_train1_feat_online):
    #用户线上消费次数：
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action == 1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_Consume_count'])
    
    #用户线上不消费次数：
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action != 1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_notConsume_count'])
    
    #用户线上点击次数，购买次数，领取次数；点击率，购买率，领取率
    df_tem_feat = df_train1_feat_online.copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_all_count'])
    
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action == 0].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_0_count'])
    
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action == 1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_1_count'])
    
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action == 2].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_action_2_count'])
    
    df_train1['on_user_action_0_rate'] = df_train1['on_user_action_0_count'] / df_train1['on_user_action_all_count']
    df_train1['on_user_action_1_rate'] = df_train1['on_user_action_1_count'] / df_train1['on_user_action_all_count']
    df_train1['on_user_action_2_rate'] = df_train1['on_user_action_2_count'] / df_train1['on_user_action_all_count']
    
    #用户线上优惠券的（领取+核销）次数：
    df_tem_feat = df_train1_feat_online[(df_train1_feat_online.Coupon_id != -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['on_user_Coupon_receive_count'])
    
    #用户线上优惠券核销的次数：
    df_tem_feat = df_train1_feat_online[(df_train1_feat_online.Coupon_id != -1) & (df_train1_feat_online.Action == 1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Action'], ['on_user_Coupon_andConsume_count'])
    
    #用户线上优惠券核销率：
    df_train1['on_user_received_Consume_rate'] = df_train1['on_user_action_Consume_count'] / (df_train1['on_user_Coupon_receive_count'] + 0.1)
    
    #用户线下不消费的次数，占线上线下总的不消费次数比重
    df_train1['on_notConsume_rate'] = df_train1['user_received_coupon_butNotConsume_count'] / \
                 (df_train1['user_received_coupon_butNotConsume_count'] + df_train1['on_user_action_notConsume_count'] + 0.1)

    #用户线下优惠券核销次数，占线上线下总的优惠券核销次数比重
    df_train1['on_coupon_Consume_rate'] = df_train1['user_received_coupon_andConsume_count'] / \
                 (df_train1['user_received_coupon_andConsume_count'] + df_train1['on_user_Coupon_andConsume_count'] + 0.1)

    #用户线上领取优惠券的次数
    df_tem_feat = df_train1_feat_online[df_train1_feat_online.Action == 2].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id'], ['Coupon_id'], ['on_user_received_coupon_count'])
    
    #用户线下领取优惠券次数，占线上线下领取次数的比重
    df_train1['on_received_coupon_rate'] = df_train1['user_received_coupon_count'] / \
                 (df_train1['user_received_coupon_count'] + df_train1['on_user_received_coupon_count'] + 0.1)
    
    return df_train1
def Merchant_feat(df_train1, df_train1_feat_offline):
    #商家优惠券发放次数
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Coupon_id != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Merchant_id'], ['Coupon_id'], ['Merchant_send_coupon_count'])
    
    #商家优惠券发放的种类
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Coupon_id != -1].copy()
    df_train1 = feat_nunique(df_train1, df_tem_feat, ['Merchant_id'], ['Coupon_id'], ['Merchant_send_coupon_nunique'])
    
    #商家优惠券发放后，不被核销的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != 1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Merchant_id'], ['Coupon_id'], ['Merchant_send_coupon_butNotConsume_count'])
    
    #商家优惠券发放后，被核销的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Merchant_id'], ['Coupon_id'], ['Merchant_send_coupon_andConsume_count'])

    #商家优惠券发放后，被核销的比率：
    df_train1['Merchant_send_coupon_andConsume_rate'] = df_train1['Merchant_send_coupon_andConsume_count'] / \
                                                        df_train1['Merchant_send_coupon_count']
    
    #商家优惠券核销的平均折率，最大折率，最小折率
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_tem_feat['Discount_rate'] = df_tem_feat.Discount_rate.apply(discount_rate_trans)
    df_train1 = feat_mean(df_train1, df_tem_feat, ['Merchant_id'], ['Discount_rate'], ['Merchant_coupon_Consume_aveDiscountRate'], na=-1)
    df_train1 = feat_max(df_train1, df_tem_feat, ['Merchant_id'], ['Discount_rate'], ['Merchant_coupon_Consume_maxDiscountRate'], na=-1)
    df_train1 = feat_min(df_train1, df_tem_feat, ['Merchant_id'], ['Discount_rate'], ['Merchant_coupon_Consume_minDiscountRate'], na=-1)
    
    #核销商家优惠券的，不同用户数量；以及占所有领取次数的比重
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_nunique(df_train1, df_tem_feat, ['Merchant_id'], ['User_id'], ['Merchant_send_coupon_userNunique'])
    df_train1['Merchant_send_coupon_userRate'] = df_train1['Merchant_send_coupon_userNunique'] / df_train1['Merchant_send_coupon_count']
    
    #商家平均每个用户核销优惠券多少张
    df_train1['Merchant_aveUser_CouponCount'] = df_train1['Merchant_send_coupon_andConsume_count'] / df_train1['Merchant_send_coupon_userNunique']
    
    #商家被核销过的不同优惠券数量；
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_nunique(df_train1, df_tem_feat, ['Merchant_id'], ['Coupon_id'], ['Merchant_send_coupon_andConsume_nunique'])
    
    
    #商家被核销过的不同优惠券数量，占所有领取过的不同优惠券比重
    df_train1['Merchant_send_coupon_andConsume_nunique'] = df_train1['Merchant_send_coupon_andConsume_nunique'] / \
                                                               df_train1['Merchant_send_coupon_nunique']
    
    #商家平均每种优惠券核销多少张：
    df_train1['Merchant_aveCoupon_count'] = df_train1['Merchant_send_coupon_andConsume_count'] / df_train1['Merchant_send_coupon_nunique']
    
    #商家被核销的优惠券中，平均，最大，最小距离。
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_mean(df_train1, df_tem_feat, ['Merchant_id'], ['Distance'], ['Merchant_send_coupon_meanDistance'], na=-1)
    df_train1 = feat_max(df_train1, df_tem_feat, ['Merchant_id'], ['Distance'], ['Merchant_send_coupon_maxDistance'], na=-1)
    df_train1 = feat_min(df_train1, df_tem_feat, ['Merchant_id'], ['Distance'], ['Merchant_send_coupon_minDistance'], na=-1)
    
    #商家被核销的优惠券中，核销的平均时间
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_tem_feat.Date = df_tem_feat.Date.apply(time_convert)
    df_tem_feat.Date_received = df_tem_feat.Date_received.apply(time_convert)
    df_tem_feat['diff'] = (df_tem_feat.Date - df_tem_feat.Date_received).dt.days
    df_train1 = feat_mean(df_train1, df_tem_feat, ['Merchant_id'], ['diff'], ['Merchant_coupon_Consume_AveTime'], na=-1)
    
    return df_train1
def user_Merchant_feat(df_train1, df_train1_feat_offline):
    #用户领取商家优惠券的次数
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Date_received != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id', 'Merchant_id'], ['Coupon_id'], ['user_get_Merchant_coupon_count'])
    
    #用户领取商家优惠券后，但不核销的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date == -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id', 'Merchant_id'], ['Coupon_id'], ['user_get_Merchant_coupon_notConsume_count'])
    
    #用户领取商家优惠券后，核销的次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Date_received != -1) & (df_train1_feat_offline.Date != -1)].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['User_id', 'Merchant_id'], ['Coupon_id'], ['user_get_Merchant_coupon_Consume_count'])
    
    #用户领取商家优惠券后，核销率
    df_train1['user_get_Merchant_coupon_Consume_rate'] = df_train1['user_get_Merchant_coupon_Consume_count'] / df_train1['user_get_Merchant_coupon_count']
    
    #用户对每个商家的不核销次数，占用户总的不核销次数的比重：
    df_train1['userMerchant_notConsume_to_user_notConsume_rate'] = df_train1['user_get_Merchant_coupon_notConsume_count'] / df_train1['user_received_coupon_butNotConsume_count']
    
    #用户对每个商家的核销次数，占用户总的核销次数的比重：
    df_train1['userMerchant_Consume_to_user_Consume_rate'] = df_train1['user_get_Merchant_coupon_Consume_count'] / df_train1['user_received_coupon_andConsume_count']
    
    #用户对每个商家的不核销次数，占商家总的不核销次数的比重：
    df_train1['userMerchant_notConsume_to_Merchant_notConsume_rate'] = df_train1['user_get_Merchant_coupon_notConsume_count'] / df_train1['Merchant_send_coupon_butNotConsume_count']
    
    #用户对每个商家的核销次数，占商家总的核销次数的比重：
    df_train1['userMerchant_Consume_to_Merchant_Consume_rate'] = df_train1['user_get_Merchant_coupon_Consume_count'] / df_train1['Merchant_send_coupon_andConsume_count']
    
    return df_train1
def Coupon_feat(df_train1, df_train1_feat_offline):
    df_train1['discount_rate_type'] = df_train1.Discount_rate.apply(lambda x: 0 if re.search(':', x) else 1)
    df_train1['discount_rate_man'] = df_train1.Discount_rate.apply(discount_rate_man)
    df_train1['discount_rate_jian'] = df_train1.Discount_rate.apply(discount_rate_jian)
    df_train1['Discount_rate'] = df_train1.Discount_rate.apply(discount_rate_trans)
    df_train1['day'] = df_train1.Date_received.dt.day
    df_train1['weekday'] = df_train1.Date_received.dt.weekday
    df_train1['is_weekend'] = df_train1.weekday.apply(lambda x: 1 if x<=4 else 0)
    
    #该优惠券在，历史上出现的次数：
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Coupon_id != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Coupon_id'], ['Merchant_id'], ['Coupon_appear_count'])
    
    #该优惠券在，历史上核销的次数：
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & df_train1_feat_offline.Date != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Coupon_id'], ['Merchant_id'], ['Coupon_Consume_count'])
    
    #该优惠券在，历史上的核销率：
    df_train1['Coupon_Consume_rate'] = df_train1['Coupon_Consume_count'] / (df_train1['Coupon_appear_count'] + 0.1)

    #历史上，该用户领取该优惠券的次数
    df_tem_feat = df_train1_feat_offline[df_train1_feat_offline.Coupon_id != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Coupon_id', 'User_id'], ['Merchant_id'], ['Coupon_User_appear_count'])
    
    #历史上，该用户领取该优惠券后的核销次数
    df_tem_feat = df_train1_feat_offline[(df_train1_feat_offline.Coupon_id != -1) & df_train1_feat_offline.Date != -1].copy()
    df_train1 = feat_count(df_train1, df_tem_feat, ['Coupon_id', 'User_id'], ['Merchant_id'], ['Coupon_User_Consume_count'])
    
    #历史上，该用户领取该优惠券后的核销率
    df_train1['Coupon_User_Consume_rate'] = df_train1['Coupon_User_Consume_count'] / (df_train1['Coupon_User_appear_count'] + 0.1)
    
    
    return df_train1
def leakage_feat(df_train1):
    #用户领取的所有优惠券数目
    df_train1 = feat_count(df_train1, df_train1, ['User_id'], ['Coupon_id'], ['leak_User_receive_all_Coupon_count'])
    
    #用户领取的相同优惠券数目
    df_train1 = feat_count(df_train1, df_train1, ['User_id', 'Coupon_id'], ['Merchant_id'], ['leak_User_receive_same_Coupon_count'])
    
    #用户领取同一优惠券的最大/最小时间： 
    df_train1_tem = df_train1.groupby(['User_id', 'Coupon_id'])['Merchant_id'].count().reset_index().rename(columns={'Merchant_id': 'count'})
    df_train1_tem = df_train1_tem[df_train1_tem['count'] >= 2]
    df_train1_tem = pd.merge(df_train1, df_train1_tem, on=['User_id', 'Coupon_id'], how='inner')
    df_train1_tem['time'] = df_train1_tem['Date_received'].dt.day
    
    df_train1 = feat_max(df_train1, df_train1_tem, ['User_id', 'Coupon_id'], ['time'], ['leak_User_receive_same_Coupon_maxTime'])
    df_train1 = feat_min(df_train1, df_train1_tem, ['User_id', 'Coupon_id'], ['time'], ['leak_User_receive_same_Coupon_minTime'])
    
    #是否是最后一次领取，是否是第一次领取：
    df_train1['is_last_receive'] = df_train1['leak_User_receive_same_Coupon_maxTime'] - (df_train1['Date_received']).dt.day
    df_train1['is_first_receive'] = (df_train1['Date_received']).dt.day - df_train1['leak_User_receive_same_Coupon_minTime']
    
    def is_firstOrLast_day(diff_day):
        if diff_day == 0:
            return 1
        elif diff_day > 0:
            return 0
        else:
            return -1
    df_train1['is_last_receive'] = df_train1['is_last_receive'].apply(is_firstOrLast_day)
    df_train1['is_first_receive'] = df_train1['is_first_receive'].apply(is_firstOrLast_day)
    
    #用户当天领取的优惠券总数：
    df_train1 = feat_count(df_train1, df_train1, ['User_id', 'Date_received'], ['Coupon_id'], ['leak_User_theDate_received_Coupon_count'])
    
    #用户当天领取相同优惠券总数：
    df_train1 = feat_count(df_train1, df_train1, ['User_id', 'Date_received', 'Coupon_id'], ['Merchant_id'], ['leak_User_theDate_received_Coupon_count'])
    
    #用户领取不同商家数目：
    df_train1 = feat_nunique(df_train1, df_train1, ['User_id'], ['Merchant_id'], ['leak_User_receive_Merchant_nunique'])
    
    #用户领取的所有优惠券种类
    df_train1 = feat_nunique(df_train1, df_train1, ['User_id'], ['Coupon_id'], ['leak_User_receive_Coupon_nunique'])

    #商家被领取的优惠券数目：
    df_train1 = feat_count(df_train1, df_train1, ['Merchant_id'], ['Coupon_id'], ['leak_Merchant_send_Coupon_count'])

    #商家被多少不用用户领取：
    df_train1 = feat_nunique(df_train1, df_train1, ['Merchant_id'], ['User_id'], ['leak_Merchant_get_User_nunique'])
    
    #商家发行的所有优惠券种类：
    df_train1 = feat_nunique(df_train1, df_train1, ['Merchant_id'], ['Coupon_id'], ['leak_Merchant_send_Coupon_nunique'])
    
    #同一张优惠券，用户这次领取与上一次/下一次领取的时间间隔：（超级强特）
    def get_day_gap_before(the_DateReceived_all_DateReceived):
        the_DateReceived, all_DateReceived = the_DateReceived_all_DateReceived.split('-')
        all_DateReceived = all_DateReceived.split(':')

        gaps = []
        for day in all_DateReceived:
            the_gap = (datetime(int(the_DateReceived[0: 4]), int(the_DateReceived[4: 6]), int(the_DateReceived[6: 8])) - 
                       datetime(int(day[0: 4]), int(day[4: 6]), int(day[6: 8]))).days
            if the_gap > 0:
                gaps.append(the_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)
    def get_day_gap_after(the_DateReceived_all_DateReceived):
        the_DateReceived, all_DateReceived = the_DateReceived_all_DateReceived.split('-')
        all_DateReceived = all_DateReceived.split(':')

        gaps = []
        for day in all_DateReceived:
            the_gap = (datetime(int(day[0: 4]), int(day[4: 6]), int(day[6: 8])) - 
                       datetime(int(the_DateReceived[0: 4]), int(the_DateReceived[4: 6]), int(the_DateReceived[6: 8]))).days
            if the_gap > 0:
                gaps.append(the_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)
    df_train1_tem = df_train1.copy()
    df_train1_tem['Date_received'] = df_train1_tem['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem = df_train1_tem.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    df_train1_tem = df_train1_tem.rename(columns={'Date_received': 'all_DateReceived'})

    df_train1_tem_2 = pd.merge(df_train1, df_train1_tem, on=['User_id', 'Coupon_id'], how='left')
    df_train1_tem_2['Date_received'] = df_train1_tem_2['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem_2['the_DateReceived_all_DateReceived'] = df_train1_tem_2.Date_received + '-' + df_train1_tem_2.all_DateReceived
    df_train1_tem_2['receive_same_Coupon_timeGap_before'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_day_gap_before)
    df_train1_tem_2['receive_same_Coupon_timeGap_after'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_day_gap_after)
    df_train1_tem_2 = df_train1_tem_2[['receive_same_Coupon_timeGap_before', 'receive_same_Coupon_timeGap_after']]
    
    df_train1 = pd.concat([df_train1.reset_index(drop=True), df_train1_tem_2.reset_index(drop=True)], axis=1)
    
    #用户这次领取与上一次/下一次领取的时间间隔：
    df_train1_tem = df_train1.copy()
    df_train1_tem['Date_received'] = df_train1_tem['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem = df_train1_tem.groupby(['User_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    df_train1_tem = df_train1_tem.rename(columns={'Date_received': 'all_DateReceived'})
    
    df_train1_tem_2 = pd.merge(df_train1, df_train1_tem, on=['User_id'], how='left')
    df_train1_tem_2['Date_received'] = df_train1_tem_2['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem_2['the_DateReceived_all_DateReceived'] = df_train1_tem_2.Date_received + '-' + df_train1_tem_2.all_DateReceived
    df_train1_tem_2['receive_Coupon_timeGap_before'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_day_gap_before)
    df_train1_tem_2['receive_Coupon_timeGap_after'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_day_gap_after)
    df_train1_tem_2 = df_train1_tem_2[['receive_Coupon_timeGap_before', 'receive_Coupon_timeGap_after']]
    
    df_train1 = pd.concat([df_train1.reset_index(drop=True), df_train1_tem_2.reset_index(drop=True)], axis=1)
    
    
    #同一张优惠券，用户此次之前/之后领取的所有优惠券数目：
    def get_Coupon_count_before(the_DateReceived_all_DateReceived):
        the_DateReceived, all_DateReceived = the_DateReceived_all_DateReceived.split('-')
        all_DateReceived = all_DateReceived.split(':')

        gaps = []
        for day in all_DateReceived:
            the_gap = (datetime(int(the_DateReceived[0: 4]), int(the_DateReceived[4: 6]), int(the_DateReceived[6: 8])) - 
                       datetime(int(day[0: 4]), int(day[4: 6]), int(day[6: 8]))).days
            if the_gap > 0:
                gaps.append(the_gap)
        if len(gaps) == 0:
            return -1
        else:
            return len(gaps)
    def get_Coupon_count_after(the_DateReceived_all_DateReceived):
        the_DateReceived, all_DateReceived = the_DateReceived_all_DateReceived.split('-')
        all_DateReceived = all_DateReceived.split(':')

        gaps = []
        for day in all_DateReceived:
            the_gap = (datetime(int(day[0: 4]), int(day[4: 6]), int(day[6: 8])) - 
                       datetime(int(the_DateReceived[0: 4]), int(the_DateReceived[4: 6]), int(the_DateReceived[6: 8]))).days
            if the_gap > 0:
                gaps.append(the_gap)
        if len(gaps) == 0:
            return -1
        else:
            return len(gaps)
    df_train1_tem = df_train1.copy()
    df_train1_tem['Date_received'] = df_train1_tem['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem = df_train1_tem.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    df_train1_tem = df_train1_tem.rename(columns={'Date_received': 'all_DateReceived'})

    df_train1_tem_2 = pd.merge(df_train1, df_train1_tem, on=['User_id', 'Coupon_id'], how='left')
    df_train1_tem_2['Date_received'] = df_train1_tem_2['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem_2['the_DateReceived_all_DateReceived'] = df_train1_tem_2.Date_received + '-' + df_train1_tem_2.all_DateReceived
    df_train1_tem_2['receive_sameCoupon_count_before'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_Coupon_count_before)
    df_train1_tem_2['receive_sameCoupon_count_after'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_Coupon_count_after)
    df_train1_tem_2 = df_train1_tem_2[['receive_sameCoupon_count_before', 'receive_sameCoupon_count_after']]
    
    df_train1 = pd.concat([df_train1.reset_index(drop=True), df_train1_tem_2.reset_index(drop=True)], axis=1)
    
    #用户此次之前/之后领取的所有优惠券数目：
    df_train1_tem = df_train1.copy()
    df_train1_tem['Date_received'] = df_train1_tem['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem = df_train1_tem.groupby(['User_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    df_train1_tem = df_train1_tem.rename(columns={'Date_received': 'all_DateReceived'})
    
    df_train1_tem_2 = pd.merge(df_train1, df_train1_tem, on=['User_id'], how='left')
    df_train1_tem_2['Date_received'] = df_train1_tem_2['Date_received'].apply(lambda x: x.strftime('%Y%m%d'))
    df_train1_tem_2['the_DateReceived_all_DateReceived'] = df_train1_tem_2.Date_received + '-' + df_train1_tem_2.all_DateReceived
    df_train1_tem_2['receive_Coupon_count_before'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_Coupon_count_before)
    df_train1_tem_2['receive_Coupon_count_after'] = df_train1_tem_2.the_DateReceived_all_DateReceived.apply(get_Coupon_count_after)
    df_train1_tem_2 = df_train1_tem_2[['receive_Coupon_count_before', 'receive_Coupon_count_after']]
    
    df_train1 = pd.concat([df_train1.reset_index(drop=True), df_train1_tem_2.reset_index(drop=True)], axis=1)

    
    return df_train1


###################################################################################
if __name__ =='__main__':
    '''
    part1：数据预处理
    '''
    ###原始数据加载#############
    on_train = pd.read_csv(input_path + 'ccf_online_stage1_train.csv', na_values='null', keep_default_na=False)
    off_train = pd.read_csv(input_path + 'ccf_offline_stage1_train.csv', na_values='null', keep_default_na=False)
    off_test = pd.read_csv(input_path + 'ccf_offline_stage1_test_revised.csv', na_values='null', keep_default_na=False)

    # 空值填-1
    on_train = on_train.fillna(-1)
    off_train = off_train.fillna(-1)
    off_test = off_test.fillna(-1)

    ###数据集划分###############
    ###测试集数据
    df_test = off_test
    df_test_feat_offline = off_train[((off_train.Date_received != -1) & (off_train.Date_received >= 20160315) & (off_train.Date_received <= 20160630)) |
                                     ((off_train.Date != -1) & (off_train.Date >= 20160315) & (off_train.Date <= 20160630))]
    df_test_feat_online = on_train[((on_train.Date_received != -1) & (on_train.Date_received >= 20160315) & (on_train.Date_received <= 20160630)) |
                                   ((on_train.Date != -1) & (on_train.Date >= 20160315) & (on_train.Date <= 20160630))]

    print('df_test.shape:', df_test.shape)

    ###训练集_1数据
    df_train1 = off_train[(off_train.Date_received != -1) & (off_train.Date_received >= 20160515) & (off_train.Date_received <= 20160615)]
    df_train1_feat_offline = off_train[((off_train.Date_received != -1) & (off_train.Date_received >= 20160201) & (off_train.Date_received <= 20160514)) |
                                       ((off_train.Date != -1) & (off_train.Date >= 20160201) & (off_train.Date <= 20160514))]
    df_train1_feat_online = on_train[((on_train.Date_received != -1) & (on_train.Date_received >= 20160201) & (on_train.Date_received <= 20160514)) |
                                     ((on_train.Date != -1) & (on_train.Date >= 20160201) & (on_train.Date <= 20160514))]

    print('df_train1.shape:', df_train1.shape)

    ###训练集_2数据
    df_train2 = off_train[(off_train.Date_received != -1) & (off_train.Date_received >= 20160414) & (off_train.Date_received <= 20160514)]
    df_train2_feat_offline = off_train[((off_train.Date_received != -1) & (off_train.Date_received >= 20160101) & (off_train.Date_received <= 20160413)) |
                                       ((off_train.Date != -1) & (off_train.Date >= 20160101) & (off_train.Date <= 20160413))]
    df_train2_feat_online = on_train[((on_train.Date_received != -1) & (on_train.Date_received >= 20160101) & (on_train.Date_received <= 20160413)) |
                                     ((on_train.Date != -1) & (on_train.Date >= 20160101) & (on_train.Date <= 20160413))]

    print('df_train2.shape:', df_train2.shape)


    ###时间格式转化、训练集打标签#####
    ###train1时间格式转化，打标签
    df_train1['Date_received'] = df_train1.Date_received.apply(time_convert)
    df_train1['Date'] = df_train1.Date.apply(time_convert)

    df_train1['diff'] = df_train1.apply(lambda x: (x.Date - x.Date_received).days if x.Date != -1 else -1, axis=1)
    df_train1['label'] = df_train1['diff'].apply(lambda x: 0 if (x > 15 or x == -1) else 1)

    print(df_train1['label'].value_counts())

    ###train2时间格式转化，打标签
    df_train2['Date_received'] = df_train2.Date_received.apply(time_convert)
    df_train2['Date'] = df_train2.Date.apply(time_convert)

    df_train2['diff'] = df_train2.apply(lambda x: (x.Date - x.Date_received).days if x.Date != -1 else -1, axis=1)
    df_train2['label'] = df_train2['diff'].apply(lambda x: 0 if (x > 15 or x == -1) else 1)

    print(df_train2['label'].value_counts())

    ###test时间格式转化
    df_test['Date_received'] = df_test.Date_received.apply(time_convert)

    print('数据预处理完毕')



    '''
    part2：特征工程
    '''
    start_time = time()
    ##########用户线下特征群；线上得分（0.5218），线下得分（0.5800）
    df_train1 = user_off_feat(df_train1, df_train1_feat_offline)
    df_train2 = user_off_feat(df_train2, df_train2_feat_offline)
    df_test = user_off_feat(df_test, df_test_feat_offline)

    ########用户线上特征群；线上得分（0.5992），线下得分（）
    df_train1 = user_on_feat(df_train1, df_train1_feat_online)
    df_train2 = user_on_feat(df_train2, df_train2_feat_online)
    df_test = user_on_feat(df_test, df_test_feat_online)

    ########商店特征群；线上得分（0.5998），线下得分（0.5781）
    df_train1 = Merchant_feat(df_train1, df_train1_feat_offline)
    df_train2 = Merchant_feat(df_train2, df_train2_feat_offline)
    df_test = Merchant_feat(df_test, df_test_feat_offline)
    
    ########用户商家交互特征群；
    df_train1 = user_Merchant_feat(df_train1, df_train1_feat_offline)
    df_train2 = user_Merchant_feat(df_train2, df_train2_feat_offline)
    df_test = user_Merchant_feat(df_test, df_test_feat_offline)

    #########优惠券特征群；线上得分（）
    df_train1 = Coupon_feat(df_train1, df_train1_feat_offline)
    df_train2 = Coupon_feat(df_train2, df_train2_feat_offline)
    df_test = Coupon_feat(df_test, df_test_feat_offline)
    
    ##########leakage特征群；线上得分（0.7683），线下得分（0.7504）
    df_train1 = leakage_feat(df_train1)
    df_train2 = leakage_feat(df_train2)
    df_test = leakage_feat(df_test)
    
    pickle.dump(df_train1, open(input_path + 'df_train1.pkl', 'wb'))
    pickle.dump(df_train2, open(input_path + 'df_train2.pkl', 'wb'))
    pickle.dump(df_test, open(input_path + 'df_test.pkl', 'wb'))
    

    print('提取特征的时间为:', int(time() - start_time))
    print("df_train1.shape:", df_train1.shape)
    print("df_train2.shape:", df_train2.shape)
    print("df_train3.shape:", df_test.shape)
    print('特征工程结束')

