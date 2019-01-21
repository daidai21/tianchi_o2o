# 阶段日志

### 阶段-1

- auc:0.52054
- 工作:复现代码
- 链接:[100行代码入门天池O2O优惠券使用新人赛【精简教程版】](https://tianchi.aliyun.com/notebook-ai/detail?postId=8462)
- 方法
  - 样本：线下数据
  - 特征：特征较少，不到十个
  - 模型：随机梯度下降SGDClassifier
- 总结
  - 特征少、模型不是最优、样本没有使用线上的
- 文件目录
  - 代码
  - 模型
  - 提交的csv文件

### 阶段-2

- auc:0.53321810
- 工作:复现代码
- 链接:[100行代码入门天池O2O优惠券使用新人赛【精简教程版】](https://tianchi.aliyun.com/notebook-ai/detail?postId=8462)
- 方法:同上 阶段-1
- 总结
  - 发现天池提交的判别有bug（提交的样例给的是要求一位小数，实际不需要）
- 文件目录
  - 代码
  - 模型
  - 提交的csv文件

### 阶段-3

- auc:
- 工作:复现代码 & 失败
- 链接:[天池 o2o 参考代码](https://tianchi.aliyun.com/notebook-ai/detail?postId=23504)
- 方法:
  - 样本：线上+线下
  - 特征：
    - 线上总共93个特征：
      - 用户特征36
      - 线下商户特征22
      - 线下优惠券特征11
      - 用户商户特征12
      - 其他特征12
    - 线下总共13个特征
  - 模型：
    - 梯度提升树 GradientBoostingClassifier 
    - xgboost XGBClassifier
    - lightgbm LGBMClassifier
    - catboost CatBoostClassifier
    - 随机森林 RandomForestClassifier
    - 极限随机树 ExtraTreesClassifier
- 总结
  - 使用了网格搜索、模型融合
  - 代码质量不认直视；一个文件32个函数；方向思路可参考
- 文件目录
  - Code-Read | 代码基本结构
  - Code
  - cache_Code.py_train.csv | 中间提取的特征csv文件

### 阶段-4

- auc:0.64030
- 工作:复现代码
- 链接:[生活大实惠：O2O优惠券使用预测](https://github.com/bike5/O2O)
- 方法:
  - 样本
  - 特征
  - 模型：Xgboost
- 总结
  - 要考虑样本分布不均；深刻理解题意；特征最重要
- 文件目录
  - 提交的csv文件
  - 第一阶段代码
  - 第二阶段代码

### 阶段-5

- auc:0.78835
- 工作:复现代码
- 链接:[python-Machine-learning/Xgboost/](https://github.com/Mryangkaitong/python-Machine-learning/tree/master/Xgboost)
- 方法:
  - 样本
  - 特征
  - 模型：xgboost
- 总结
  - xgboost性能还不错
- 文件目录
  - Data_Preprocessed | 预处理过的数据
  - Feature-Produce.ipynb | 提取特征
  - Xgboost.ipynb | 模型训练
  - 自己复现&修改
    - ProcessDataSet1.csv | 预处理后数据集
    - ProcessDataSet2.csv | 预处理后数据集
    - ProcessDataSet3.csv | 预处理后数据集
    - xgb_feature_score.csv | 特征分数
    - xgbmodel | 模型
    - xgbmodel1 | 模型
    - xgb_preds.csv | 预测的数据（这里提交的csv文件是xgb_preds.csv）
    - xgb_preds2.csv | 预测的数据
    - xgboost-复现.py
    - 复现xgboost--预测.py

### 阶段-6

- auc:0.78525997
- 工作:复现代码
- 链接:[python-Machine-learning/Xgboost/](https://github.com/Mryangkaitong/python-Machine-learning/tree/master/Xgboost)
- 方法:
- 总结:
  - xgboost性能还不错；训练样本貌似过拟合了
- 文件目录
  - 同上 阶段-5（这里提交的csv文件是xgb_preds2.csv）

### 阶段-7
