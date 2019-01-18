import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix

'''
from keras.layers import Input, Embedding, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras import optimizers
'''


###编码，one-hot，归一化
def encode_count(df, encoder_list):
    lbl = LabelEncoder()
    for i in range(0, len(encoder_list)):
        str_column_name = encoder_list[i]
        df[[str_column_name]] = lbl.fit_transform(df[[str_column_name]])
    return df
def encode_onehot(df, oneHot_list):
    for i in range(0, len(oneHot_list)):
        str_column_name = oneHot_list[i]
        feature_df = pd.get_dummies(df[str_column_name], prefix=str_column_name)
        df = pd.concat([df.drop([str_column_name], axis=1), feature_df], axis=1)
    return df
def normalize(df, normalize_list):
    scaler = StandardScaler()
    for i in range(0, len(normalize_list)):
        str_column_name = normalize_list[i]
        df[[str_column_name]] = scaler.fit_transform(df[[str_column_name]])
    return df

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

##特征选择
def RFECV_feature_sel(X_train, y_train, X_val, X_test):  
    ##天池的大佬经常使用！！！
    ###clf可以随便换！！！
    clf = lgb.LGBMClassifier()
    selor = RFECV(clf, step=1, cv=3)
    selor = selor.fit(X_train, y_train)

    X_train_sel = selor.transform(X_train)
    X_val_sel = selor.transform(X_val)
    X_test_sel = selor.transform(X_test)
    
    return selor, X_train_sel, X_val_sel, X_test_sel
def Tree_feature_sel(X_train, y_train, X_val, y_val, X_test, sel_num): 
     ##最简单最常用
    clf = lgb.LGBMClassifier(n_estimators=10000,
                             learning_rate=0.06,
                             max_depth=5,
                             num_leaves=30,
                             objective='binary',
                             subsample=0.9,
                             sub_feature=0.9,
                            )
    clf = clf.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
                  early_stopping_rounds=100, verbose = 500, 
                 )

    if type(X_train) is pd.core.frame.DataFrame:
        print('type(X_train) is DataFrame')
        feat_impo = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
        sel_list = [feat[0] for feat in feat_impo[0: sel_num]]
        X_train_sel = X_train[sel_list]
        X_val_sel = X_val[sel_list]
        X_test_sel = X_test[sel_list]
    else:
        if type(X_train) is np.ndarray:
            print('type(X_train) is ndarray')
            feat_impo = sorted(zip(range(0, len(X_train[0])), clf.feature_importances_), key=lambda x: x[1], reverse=True)
            
        elif type(X_train) is csr_matrix:
            print('type(X_train) is csr_matrix')
            feat_impo = sorted(zip(range(0, X_train.get_shape()[1]), clf.feature_importances_), key=lambda x: x[1], reverse=True)
        else:
            print('暂时不支持')
            return

        sel_list = [feat[0] for feat in feat_impo[0: sel_num]]
        X_train_sel = X_train[:, sel_list]
        X_val_sel = X_val[:, sel_list]
        X_test_sel = X_test[sel_list]
        
    print("sel_num is:", sel_num)
    return feat_impo, X_train_sel, X_val_sel, X_test_sel


##文本分类textcnn
def clf_text_cnn(maxlen, max_features, embed_size):
    #maxlen:句子的最大长度
    #max_features:词典中单词的个数，（每个单词可以看成是一个特征，因此叫最大特征）
    #embed_size:将一个单词映射成多少维的向量
    
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(30, activation='relu')(out)   #卷积输出的特征

    output = Dense(units=4, activation='softmax')(output)
    
    model = Model([comment_seq], output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) 
    return model