2019-01-23 05:12:39
grid_search_auto
grid_search
--------------------------------------------
2019-01-23 05:12:39
{'n_estimators': array([1800, 1900, 2000])}

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
time: 0:03:21
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
6,^[[Dtask end
task end
task end
task end
task end
task end
task end
time: 0:07:17
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
/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
  warnings.warn(CV_WARNING, FutureWarning)
^CTraceback (most recent call last):
  File "code-gdbt.py", line 1240, in <module>
    grid_search_gbdt()  # 网格搜索gdbt模型
  File "code-gdbt.py", line 1090, in grid_search_gbdt
    grid_search_auto(steps, params, GradientBoostingClassifier())
  File "code-gdbt.py", line 1031, in grid_search_auto
    best_params, best_score = grid_search(estimator.set_params(**params), param_grid)
  File "code-gdbt.py", line 981, in grid_search
    clf = fit_eval_metric(clf, X, y, estimator_name)
  File "code-gdbt.py", line 939, in fit_eval_metric
    estimator.fit(X, y)
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 722, in fit
    self._run_search(evaluate_candidates)
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 1191, in _run_search
    evaluate_candidates(ParameterGrid(self.param_grid))
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 711, in evaluate_candidates
    cv.split(X, y, groups)))
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/parallel.py", line 930, in __call__
    self.retrieve()
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/parallel.py", line 833, in retrieve
    self._output.extend(job.get(timeout=self.timeout))
  File "/Users/daidai/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 521, in wrap_future_result
    return future.result(timeout=timeout)
  File "/Users/daidai/anaconda3/lib/python3.7/concurrent/futures/_base.py", line 427, in result
    self._condition.wait(timeout)
  File "/Users/daidai/anaconda3/lib/python3.7/threading.py", line 296, in wait
    waiter.acquire()
KeyboardInterrupt
