2019-01-22 16:20:11
grid_search_xgb
grid_search_auto
grid_search
--------------------------------------------
2019-01-22 16:20:11
{'n_estimators': array([1250, 1260, 1270])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
^Pfit_eval_metric end
0.89927 (+/-0.00505) for {'n_estimators': 1250}
0.89927 (+/-0.00507) for {'n_estimators': 1260}
0.89928 (+/-0.00507) for {'n_estimators': 1270}

best params {'n_estimators': 1270}
best score 0.8992800136276925
time: 0:24:09

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 16:44:20
{'n_estimators': array([1280])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89927 (+/-0.00508) for {'n_estimators': 1280}

best params {'n_estimators': 1280}
best score 0.8992721170348597
time: 0:10:22

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 16:54:43
{'max_depth': array([7, 8, 9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89888 (+/-0.00491) for {'max_depth': 7}
0.89928 (+/-0.00507) for {'max_depth': 8}
0.89918 (+/-0.00494) for {'max_depth': 9}

best params {'max_depth': 8}
best score 0.8992800136276925
time: 0:24:03

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 17:18:47
{'min_child_weight': array([3, 4, 5])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89899 (+/-0.00487) for {'min_child_weight': 3}
0.89928 (+/-0.00507) for {'min_child_weight': 4}
0.89912 (+/-0.00491) for {'min_child_weight': 5}

best params {'min_child_weight': 4}
best score 0.8992800136276925
time: 0:23:41

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 17:42:28
{'gamma': array([0.1, 0.2, 0.3])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89921 (+/-0.00496) for {'gamma': 0.1}
0.89928 (+/-0.00507) for {'gamma': 0.2}
0.89919 (+/-0.00487) for {'gamma': 0.30000000000000004}

best params {'gamma': 0.2}
best score 0.8992800136276925
time: 0:23:43

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 18:06:12
{'subsample': array([0.5, 0.6, 0.7])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89887 (+/-0.00509) for {'subsample': 0.5}
0.89928 (+/-0.00507) for {'subsample': 0.6}
0.89939 (+/-0.00472) for {'subsample': 0.7}

best params {'subsample': 0.7}
best score 0.8993860900649566
time: 0:23:33

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 18:29:45
{'subsample': array([0.8])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89968 (+/-0.00418) for {'subsample': 0.7999999999999999}

best params {'subsample': 0.7999999999999999}
best score 0.8996808728496932
time: 0:10:26

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 18:40:12
{'subsample': array([0.9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89953 (+/-0.00475) for {'subsample': 0.8999999999999999}

best params {'subsample': 0.8999999999999999}
best score 0.899529311190782
time: 0:33:09

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 19:13:22
{'colsample_bytree': array([0.7, 0.8, 0.9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
^[[Dfit_eval_metric end
0.89956 (+/-0.00467) for {'colsample_bytree': 0.7000000000000001}
0.89968 (+/-0.00418) for {'colsample_bytree': 0.8}
0.89950 (+/-0.00436) for {'colsample_bytree': 0.9}

best params {'colsample_bytree': 0.8}
best score 0.8996808728496932
time: 0:24:19

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 19:37:42
{'scale_pos_weight': array([1, 2])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89968 (+/-0.00418) for {'scale_pos_weight': 1}
0.89884 (+/-0.00439) for {'scale_pos_weight': 2}

best params {'scale_pos_weight': 1}
best score 0.8996808728496932
time: 0:33:14

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 20:10:57
{'reg_alpha': array([0. , 0.1])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89968 (+/-0.00418) for {'reg_alpha': 0.0}
0.89968 (+/-0.00454) for {'reg_alpha': 0.1}

best params {'reg_alpha': 0.1}
best score 0.8996842899696718
time: 0:17:20

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 20:28:17
{'reg_alpha': array([0.2])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89961 (+/-0.00442) for {'reg_alpha': 0.2}

best params {'reg_alpha': 0.2}
best score 0.8996113403607712
time: 0:10:26

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1270, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
--------------------------------------------
new grid search
grid_search
--------------------------------------------
2019-01-22 20:38:44
{'n_estimators': array([1260, 1270, 1280])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89967 (+/-0.00456) for {'n_estimators': 1260}
0.89968 (+/-0.00454) for {'n_estimators': 1270}
0.89971 (+/-0.00455) for {'n_estimators': 1280}

best params {'n_estimators': 1280}
best score 0.8997062593915979
time: 0:24:33

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1280, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 21:03:18
{'n_estimators': array([1290])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89972 (+/-0.00455) for {'n_estimators': 1290}

best params {'n_estimators': 1290}
best score 0.8997159790322135
time: 0:10:36

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 21:13:55
{'n_estimators': array([1300])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89972 (+/-0.00456) for {'n_estimators': 1300}

best params {'n_estimators': 1300}
best score 0.8997152679853041
time: 0:10:45

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 21:24:40
{'max_depth': array([7, 8, 9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89939 (+/-0.00424) for {'max_depth': 7}
0.89972 (+/-0.00455) for {'max_depth': 8}
0.89932 (+/-0.00452) for {'max_depth': 9}

best params {'max_depth': 8}
best score 0.8997159790322135
time: 0:25:03

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 21:49:44
{'min_child_weight': array([3, 4, 5])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89962 (+/-0.00443) for {'min_child_weight': 3}
0.89972 (+/-0.00455) for {'min_child_weight': 4}
0.89968 (+/-0.00484) for {'min_child_weight': 5}

best params {'min_child_weight': 4}
best score 0.8997159790322135
time: 0:24:50

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 22:14:34
{'gamma': array([0.1, 0.2, 0.3])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89981 (+/-0.00437) for {'gamma': 0.1}
0.89972 (+/-0.00455) for {'gamma': 0.2}
0.89971 (+/-0.00423) for {'gamma': 0.30000000000000004}

best params {'gamma': 0.1}
best score 0.899814704200253
time: 0:25:01

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 22:39:35
{'gamma': array([0.])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89968 (+/-0.00446) for {'gamma': 0.0}

best params {'gamma': 0.0}
best score 0.8996752655082665
time: 0:10:35

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 22:50:11
{'subsample': array([0.7, 0.8, 0.9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89938 (+/-0.00490) for {'subsample': 0.7}
0.89981 (+/-0.00437) for {'subsample': 0.7999999999999999}
0.89952 (+/-0.00456) for {'subsample': 0.8999999999999999}

best params {'subsample': 0.7999999999999999}
best score 0.899814704200253
time: 0:24:43

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 23:14:54
{'colsample_bytree': array([0.7, 0.8, 0.9])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89962 (+/-0.00485) for {'colsample_bytree': 0.7000000000000001}
0.89981 (+/-0.00437) for {'colsample_bytree': 0.8}
0.89956 (+/-0.00440) for {'colsample_bytree': 0.9}

best params {'colsample_bytree': 0.8}
best score 0.899814704200253
time: 0:23:59

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 23:38:54
{'scale_pos_weight': array([1, 2])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89981 (+/-0.00437) for {'scale_pos_weight': 1}
0.89877 (+/-0.00425) for {'scale_pos_weight': 2}

best params {'scale_pos_weight': 1}
best score 0.899814704200253
time: 0:17:07

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
grid_search
--------------------------------------------
2019-01-22 23:56:01
{'reg_alpha': array([0. , 0.1, 0.2])}

get_train_data
get_train_data | end
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
fit_eval_metric
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
fit_eval_metric end
0.89965 (+/-0.00444) for {'reg_alpha': 0.0}
0.89981 (+/-0.00437) for {'reg_alpha': 0.1}
0.89967 (+/-0.00456) for {'reg_alpha': 0.2}

best params {'reg_alpha': 0.1}
best score 0.899814704200253
time: 0:29:00

grid_search
XGBClassifier {'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
--------------------------------------------
grid_search_auto end
grid_search_xgb end
2019-01-22 16:20:11
grid search: XGBClassifier
{'learning_rate': 0.01, 'n_estimators': 1290, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0.1, 'subsample': 0.7999999999999999, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'reg_alpha': 0.1, 'n_jobs': 7, 'seed': 0}
time: 8:04:50
----------------------------------------------------
