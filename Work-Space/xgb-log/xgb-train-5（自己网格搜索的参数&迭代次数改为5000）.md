2019-01-23 04:29:53
train_xgb
grid_search_xgb
train
get_train_data
get_preprocess_data
get_preprocess_data end
# ==========线下特征 开始==========
132943 11
# ==========用户特征==========
# ==========商户特征==========
# ==========优惠券特征==========
# ==========用户-商户特征==========
# ==========其他特征==========
task
0,task
1,task
2,task
3,task
4,task
5,task
6,task end
task end
task end
task end
task end
task end
task end
time: 0:03:31
132943 111
# ==========线下特征 结束==========
# ==========线上特征 开始==========
132943 124
----------
# ==========线上特征 结束==========
# ==========线下特征 开始==========
252586 11
# ==========用户特征==========
# ==========商户特征==========
# ==========优惠券特征==========
# ==========用户-商户特征==========
# ==========其他特征==========
task
0,task
1,task
2,task
3,task
4,task
5,task
6,task end
task end
task end
task end
task end
task end
task end
time: 0:07:14
252586 111
# ==========线下特征 结束==========
# ==========线上特征 开始==========
252586 124
----------
# ==========线上特征 结束==========
drop_columns
drop_columns
get_train_data end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
fit_eval_metric end
train end
train_xgb end
2019-01-23 04:29:53
xgb, learning_rate: 0.010, n_estimators: 5000, max_depth: 8, min_child_weight: 4, gamma: 0.1, subsample: 0.8, colsample_bytree: 0.8

  accuracy: 0.936275
       auc: 0.897815
coupon auc: 0.779099

time: 0:27:57
----------------------------------------------------
