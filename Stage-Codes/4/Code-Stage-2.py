# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:07:14 2016

@author: Administrator
"""
import csv
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import auc,roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


    
def atrian(trainData,labels):
    
    trainData = np.array(trainData); labels = np.array(labels);
    dtrain = xgb.DMatrix(trainData,labels)
    params = {'max_depth':6, 'eta':0.2,'silent':1, 'objective':'binary:logistic','eval_metric':'auc'}
    """    
    params = {
            'booster':'gbtree',
            'objective':'binary:logistic',
            'eta':0.1,
            'max_depth':10,
            'subsample':0.8,
            'min_child_weight':5,
            'colsample_bytree':0.2,
            'scale_pos_weight':0.1,
            'eval_metric':'auc',
            'gamma':0.2,            
            
    }
    """
    num_round = 30
    bst = xgb.train(params,dtrain,num_round)
    #bst = LogisticRegression()
    #bst.fit(np.array(trainData),np.array(labels))
    return bst
    
def predict(bst, testData):
    dtest = xgb.DMatrix(testData)
    prob = list(bst.predict(dtest))
    """
    probArr = bst.predict_proba(testData); prob = []
    m = len(probArr)
    for i in xrange(m):
        prob.append(probArr[i][1])
    """
    return prob


    
def submitFile(offTestData,prob):
    m = len(prob)

    with open("GBDT_7136.csv","wb+") as fs:
        myWrite = csv.writer(fs)
        for i in xrange(m):
            myWrite.writerow([offTestData[i][0],offTestData[i][2],offTestData[i][-1],prob[i]])
            
            
def merge(trainData2,trainData3,trainData4,trainData5,label2,label3,label4,label5):
    trainData = []; 
    label = label2+label3+label4+label5
    for line in trainData2:
        trainData.append(line)
    for line in trainData3:
        trainData.append(line)
    for line in trainData4:
        trainData.append(line)
    for line in trainData5:
        trainData.append(line)

    return trainData,label


#计算AUC值
def computeAuc(prob,label,coupon):
    from sklearn.metrics import roc_auc_score
    dic = {}
    num_of_coupons = 0
    for i, c in enumerate(coupon):
        if c not in dic:
            dic[c] = ([],[])
            num_of_coupons += 1
        dic[c][0].append(label[i])
        dic[c][1].append(prob[i])
    score = 0.0
    num_of_one_label_record = 0
    num_of_one_label_coupon = 0
    num_of_valid_coupons = 0
    for c in dic:
        if len(set(dic[c][0])) == 1:
            num_of_one_label_record += len(dic[c][0])
            num_of_one_label_coupon += 1
            continue
        score += roc_auc_score(dic[c][0],dic[c][1])
        num_of_valid_coupons += 1
    print num_of_coupons,num_of_valid_coupons,num_of_one_label_coupon,len(label),num_of_one_label_record
    print score / num_of_valid_coupons

#归一化
def normal(trainData,testData):
    min_max_scaler = preprocessing.MinMaxScaler()
    trainData = min_max_scaler.fit_transform(trainData)
    testData = min_max_scaler.fit_transform(testData)
    
    return trainData,testData

###线下评测
def metricsPro(trainData_2345,trainData_6,label_2345, label6, coupon6):
    x_train = trainData_2345; x_test = trainData_6; y_train = label_2345; y_test = label6
    """
    bst = LogisticRegression()
    bst.fit(np.array(x_train),np.array(y_train))
    probArr = bst.predict_proba(x_test); prob = []
    m = len(probArr)
    for i in xrange(m):
        prob.append(probArr[i][1])
    """
    dtrain = xgb.DMatrix(x_train,y_train)
    params = {'max_depth':6, 'eta':0.2,'silent':1, 'objective':'binary:logistic','eval_metric':'auc'}    
    dtest = xgb.DMatrix(x_test,y_test)
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    num_round = 40
    bst = xgb.train(params,dtrain,num_round,watchlist)
    prob = list(bst.predict(dtest))
    
    computeAuc(prob,y_test,coupon6)
    
    return bst

def metricsProAll(testData, trainData, label, label6, coupon6):
    x_train = trainData; x_test = testData; y_train = label; y_test = label6
    
    '''
    bst = atrian(x_train,y_train)
    probArr = bst.predict_proba(x_test); prob = []
    m = len(probArr)
    for i in xrange(m):
        prob.append(probArr[i][1])
    '''
    dtrain = xgb.DMatrix(x_train,y_train)
    params = {'max_depth':6, 'eta':0.2,'silent':1, 'objective':'binary:logistic','eval_metric':'auc'}    
    dtest = xgb.DMatrix(x_test,y_test)
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    num_round = 100
    bst = xgb.train(params,dtrain,num_round,watchlist)
    prob = list(bst.predict(dtest))
    
    computeAuc(prob,y_test,coupon6)
    
    return bst  
    

def importLabel():
    with open('train_test_label_data\label.csv') as fs:
        data = csv.reader(fs)
        for line in data:
            label = map(int,line)
    return label
    
def importData():
    trainData = []; testData = []
    with open(r"train_test_label_data\trainData_all_month.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            trainData.append(map(int,line))
            
    with open(r"train_test_label_data\testData_all_month.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            testData.append(map(int,line))
    return trainData, testData    
    

    
##分割训练集数据
def splitData():
    data1 = []; data2 = []; data3 = []; data4 = []; data5 = []; data6 = []; data12 = [];data123 = [];data1234=[]
    data12345 = []
    with open("ccf_offline_stage1_train.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            if line[5][4:6] == '01' or (line[5]=='null' and line[6][4:6] == '01'):
                data1.append(line)
            if line[5][4:6] == '02' or (line[5]=='null' and line[6][4:6] == '02'):
                data2.append(line)
            if line[5][4:6] == '03' or (line[5]=='null' and line[6][4:6] == '03'):
                data3.append(line)
            if line[5][4:6] == '04' or (line[5]=='null' and line[6][4:6] == '04'):
                data4.append(line)
            if line[5][4:6] == '05' or (line[5]=='null' and line[6][4:6] == '05'):
                data5.append(line)
            if line[5][4:6] == '06' or (line[5]=='null' and line[6][4:6] == '06'):
                data6.append(line)
            if (line[5][4:6] == '01' or (line[5]=='null' and line[6][4:6] == '01')) or (line[5][4:6] == '02' or (line[5]=='null' and line[6][4:6] == '02')):
                data12.append(line)
            if (line[5][4:6] == '01' or (line[5]=='null' and line[6][4:6] == '01')) or (line[5][4:6] == '02' or (line[5]=='null' and line[6][4:6] == '02'))\
            or (line[5][4:6] == '03' or (line[5]=='null' and line[6][4:6] == '03')):
                data123.append(line)
            if (line[5][4:6] == '01' or (line[5]=='null' and line[6][4:6] == '01')) or (line[5][4:6] == '02' or (line[5]=='null' and line[6][4:6] == '02'))\
            or (line[5][4:6] == '03' or (line[5]=='null' and line[6][4:6] == '03')) or (line[5][4:6] == '04' or (line[5]=='null' and line[6][4:6] == '04')):
                data1234.append(line)
            if (line[5][4:6] == '01' or (line[5]=='null' and line[6][4:6] == '01')) or (line[5][4:6] == '02' or (line[5]=='null' and line[6][4:6] == '02'))\
            or (line[5][4:6] == '03' or (line[5]=='null' and line[6][4:6] == '03')) or (line[5][4:6] == '04' or (line[5]=='null' and line[6][4:6] == '04'))\
            or (line[5][4:6] == '05' or (line[5]=='null' and line[6][4:6] == '05')):
                data12345.append(line)
            
    with open('splitData\data1.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data1:
            myWrite.writerow(line)
    with open('splitData\data2.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data2:
            myWrite.writerow(line)
    with open('splitData\data3.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data3:
            myWrite.writerow(line)
    with open('splitData\data4.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data4:
            myWrite.writerow(line)
    with open('splitData\data5.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data5:
            myWrite.writerow(line)
    with open('splitData\data6.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data6:
            myWrite.writerow(line)
    with open('splitData\data12.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data12:
            myWrite.writerow(line)
    with open('splitData\data123.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data123:
            myWrite.writerow(line)
    with open('splitData\data1234.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data1234:
            myWrite.writerow(line)
    with open('splitData\data12345.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in data12345:
            myWrite.writerow(line)
   
