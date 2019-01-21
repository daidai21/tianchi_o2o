# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:49:30 2016

@author: Administrator
"""
from __future__ import division
import numpy as np
import csv
from datetime import datetime as Date
import time

# 用来提交时的数据
def getOffLineTestUser():
    offTestData = []
    with open("ccf_offline_stage1_test_revised.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            offTestData.append(line)
    return offTestData

##统计使用情况
#当前月之前所有的月份的统计信息
def getAllMonth():
    #训练数据
    otherF2_1 = one_condition('splitData\data1.csv','splitData\data2.csv')
    otherF2_2 = two_condition('splitData\data1.csv','splitData\data2.csv')
    otherF2_3 = three_condition('splitData\data1.csv','splitData\data2.csv')
    #otherF2_4 = four_condition('splitData\data1.csv','splitData\data2.csv')
    #线上特征
    otherF1 = onlineData('onlineData\data1.csv','splitData\data2.csv')   
    
    otherF3_1 = one_condition('splitData\data12.csv','splitData\data3.csv')
    otherF3_2 = two_condition('splitData\data12.csv','splitData\data3.csv')
    otherF3_3 = three_condition('splitData\data12.csv','splitData\data3.csv')
    #otherF3_4 = four_condition('splitData\data12.csv','splitData\data3.csv')
    #线上特征
    otherF2 = onlineData('onlineData\data12.csv','splitData\data3.csv')     
    
    otherF4_1 = one_condition('splitData\data123.csv','splitData\data4.csv')
    otherF4_2 = two_condition('splitData\data123.csv','splitData\data4.csv')
    otherF4_3 = three_condition('splitData\data123.csv','splitData\data4.csv')
    #otherF4_4 = four_condition('splitData\data123.csv','splitData\data4.csv')
    #线上特征
    otherF3 = onlineData('onlineData\data123.csv','splitData\data4.csv') 
    
    
    otherF5_1 = one_condition('splitData\data1234.csv','splitData\data5.csv')
    otherF5_2 = two_condition('splitData\data1234.csv','splitData\data5.csv')    
    otherF5_3 = three_condition('splitData\data1234.csv','splitData\data5.csv')
    #otherF5_4 = four_condition('splitData\data1234.csv','splitData\data5.csv')
    #线上特征
    otherF4 = onlineData('onlineData\data1234.csv','splitData\data5.csv') 
    
    otherF6_1 = one_condition('splitData\data12345.csv','splitData\data6.csv')
    otherF6_2 = two_condition('splitData\data12345.csv','splitData\data6.csv')
    otherF6_3 = three_condition('splitData\data12345.csv','splitData\data6.csv')
    #otherF6_4 = four_condition('splitData\data12345.csv','splitData\data6.csv')
    #线上特征
    otherF5 = onlineData('onlineData\data12345.csv','splitData\data6.csv') 
    #####      当月券情况
    
    otherF2_online = onlineNum('splitData\data2.csv','splitData\data2.csv')
    otherF3_online = onlineNum('splitData\data3.csv','splitData\data3.csv')
    otherF4_online = onlineNum('splitData\data4.csv','splitData\data4.csv')
    otherF5_online = onlineNum('splitData\data5.csv','splitData\data5.csv')
    otherF6_online = onlineNum('splitData\data6.csv','splitData\data6.csv')
    
    trainData_2 = getData(otherF2_1, otherF2_2, otherF2_3, otherF2_online,otherF1) 
    trainData_3 = getData(otherF3_1, otherF3_2, otherF3_3, otherF3_online,otherF2)
    trainData_4 = getData(otherF4_1, otherF4_2, otherF4_3, otherF4_online,otherF3)
    trainData_5 = getData(otherF5_1, otherF5_2, otherF5_3, otherF5_online,otherF4)
    trainData_6 = getData(otherF6_1, otherF6_2, otherF6_3, otherF6_online,otherF5)
    
    trainData = merge(trainData_2, trainData_3, trainData_4, trainData_5, trainData_6)
    
    label2, coupon2 = getLabels('splitData\data2.csv')
    label3, coupon3 = getLabels('splitData\data3.csv')
    label4, coupon4 = getLabels('splitData\data4.csv')
    label5, coupon5 = getLabels('splitData\data5.csv')
    label6, coupon6 = getLabels('splitData\data6.csv')
    
    label = label2 + label3 + label4 + label5 + label6
    
    label_2345 = label2 + label3 + label4 + label5
    trainData_2345 = merge2345(trainData_2, trainData_3, trainData_4, trainData_5)
    
    #测试数据
    otherF7_1 = one_condition('splitData\data123456.csv','splitData\data7.csv')
    otherF7_2 = two_condition('splitData\data123456.csv','splitData\data7.csv')
    otherF7_3 = three_condition('splitData\data123456.csv','splitData\data7.csv')
    #otherF7_4 = four_condition('splitData\data123456.csv','splitData\data7.csv')
    #当月特征
    otherF7_online = onlineNum('splitData\data7.csv','splitData\data7.csv')
    #线上特征
    otherF6 = onlineData('onlineData\data123456.csv','splitData\data7.csv')     
    
    testData = getData(otherF7_1, otherF7_2, otherF7_3,otherF7_online,otherF6)
    
    return trainData, testData, label, trainData_2345, trainData_6, label_2345, label6, coupon6

#仅当前月之前一个月的统计信息
def getEveryMonth():
    #训练数据
    otherF2_1 = one_condition('splitData\data1.csv','splitData\data2.csv')
    otherF2_2 = two_condition('splitData\data1.csv','splitData\data2.csv')
    otherF2_3 = three_condition('splitData\data1.csv','splitData\data2.csv')
    #otherF2_4 = four_condition('splitData\data1.csv','splitData\data2.csv')
    
    otherF3_1 = one_condition('splitData\data2.csv','splitData\data3.csv')
    otherF3_2 = two_condition('splitData\data2.csv','splitData\data3.csv')
    otherF3_3 = three_condition('splitData\data2.csv','splitData\data3.csv')
    #otherF3_4 = four_condition('splitData\data2.csv','splitData\data3.csv')
    
    otherF4_1 = one_condition('splitData\data3.csv','splitData\data4.csv')
    otherF4_2 = two_condition('splitData\data3.csv','splitData\data4.csv')
    otherF4_3 = three_condition('splitData\data3.csv','splitData\data4.csv')
    #otherF4_4 = four_condition('splitData\data3.csv','splitData\data4.csv')
    
    
    otherF5_1 = one_condition('splitData\data4.csv','splitData\data5.csv')
    otherF5_2 = two_condition('splitData\data4.csv','splitData\data5.csv')    
    otherF5_3 = three_condition('splitData\data4.csv','splitData\data5.csv')
    #otherF5_4 = four_condition('splitData\data4.csv','splitData\data5.csv')
    
    otherF6_1 = one_condition('splitData\data5.csv','splitData\data6.csv')
    otherF6_2 = two_condition('splitData\data5.csv','splitData\data6.csv')
    otherF6_3 = three_condition('splitData\data5.csv','splitData\data6.csv')
    #otherF6_4 = four_condition('splitData\data5.csv','splitData\data6.csv')
    
    #####      当月券情况
    
    otherF2_online = onlineNum('splitData\data2.csv','splitData\data2.csv')
    otherF3_online = onlineNum('splitData\data3.csv','splitData\data3.csv')
    otherF4_online = onlineNum('splitData\data4.csv','splitData\data4.csv')
    otherF5_online = onlineNum('splitData\data5.csv','splitData\data5.csv')
    otherF6_online = onlineNum('splitData\data6.csv','splitData\data6.csv')
    
    trainData_2 = getData(otherF2_1,otherF2_2, otherF2_3,otherF2_online) 
    trainData_3 = getData(otherF3_1,otherF3_2, otherF3_3,otherF3_online)
    trainData_4 = getData(otherF4_1,otherF4_2, otherF4_3,otherF4_online)
    trainData_5 = getData(otherF5_1,otherF5_2, otherF5_3,otherF5_online)
    trainData_6 = getData(otherF6_1, otherF6_3,otherF6_2,otherF6_online)
    
    trainData = merge(trainData_2, trainData_3, trainData_4, trainData_5, trainData_6)
    
    label2, coupon2 = getLabels('splitData\data2.csv')
    label3, coupon3 = getLabels('splitData\data3.csv')
    label4, coupon4 = getLabels('splitData\data4.csv')
    label5, coupon5 = getLabels('splitData\data5.csv')
    label6, coupon6 = getLabels('splitData\data6.csv')
    
    label = label2 + label3 + label4 + label5 + label6
    
    label_2345 = label2 + label3 + label4 + label5
    trainData_2345 = merge2345(trainData_2, trainData_3, trainData_4, trainData_5)
    
    #测试数据
    otherF7_1 = one_condition('splitData\data6.csv','splitData\data7.csv')
    otherF7_2 = two_condition('splitData\data6.csv','splitData\data7.csv')
    otherF7_3 = three_condition('splitData\data6.csv','splitData\data7.csv')
    #otherF7_4 = four_condition('splitData\data6.csv','splitData\data7.csv')
    otherF7_online = onlineNum('splitData\data7.csv','splitData\data7.csv')
    
    testData = getData(otherF7_1,otherF7_2, otherF7_3,otherF7_online)
    
    return trainData, testData, label, trainData_2345, trainData_6, label_2345, label6, coupon6

    
    
def merge2345(trainData_2, trainData_3, trainData_4, trainData_5):
    trainData2345 = []
    for line in trainData_2:
        trainData2345.append(line)
    for line in trainData_3:
        trainData2345.append(line)
    for line in trainData_4:
        trainData2345.append(line)
    for line in trainData_5:
        trainData2345.append(line)
    
    return trainData2345
    
#train数据打上标签，提取每条数据的券id（为了线下测AUC）
def getLabels(filename):
    labels = []; coupon = []
    with open(filename) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null' :
                if line[-1] != 'null' and (Date.strptime(line[-1], '%Y%m%d')-Date.strptime(line[-2], '%Y%m%d')).days<15:
                    labels.append(1)
                    coupon.append(line[2])
                else:
                    labels.append(0)
                    coupon.append(line[2])

    return labels,coupon

##线上特征
def onlineData(filename1,filename2):
    user_click = {}; user_c = {}; user_c_consum = {}
    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] == '0':
                #用户点击次数
                user_click[line[0]] = user_click.get(line[0],0) + 1
            if line[2] == '2':                
                #用户领券次数
                user_c[line[0]] = user_c.get(line[0],0) + 1
            if line[2] == '2' and line[-1] != 'null':
                #用户领券消费次数
                user_c_consum[line[0]] = user_c_consum.get(line[0],0) + 1


    otherF = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                otherF.append([user_click.get(line[0],0),user_c.get(line[0],0),user_c_consum.get(line[0],0)])
        
    return otherF
  
    
##统计当月信息（函数名字乱取）
def onlineNum(filename1,filename2):
    user_c_get_num={}; c_num = {} ; b_num = {}; b_c_num = {}; user_b = {}; user_dis = {}; b_dis={};c_dis={};dis={}
    user_c_dis = {};   user_b_dis={} ;b_c_dis = {}; user_get_num = {}
    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                #该人当月领了多少张该券
                user_c_get_num[line[0]+line[2]] = user_c_get_num.get(line[0]+line[2],0) + 1
                #该人当月领了多少张券                
                user_get_num[line[0]] = user_get_num.get(line[0],0) + 1
                #该券当月发了多少张
                c_num[line[2]] = c_num.get(line[2], 0) + 1
                #该商户发放多少张券
                b_num[line[1]] = b_num.get(line[1],0) + 1
                #该商户发了该券多少张
                b_c_num[line[1]+line[2]] = b_c_num.get(line[1]+line[2],0) + 1
                #该用户领了该店多少次券
                user_b[line[0]+line[1]] = user_b.get(line[0]+line[1],0) + 1
                #该用户领了该折扣的券多少次
                user_dis[line[0]+line[3]] = user_dis.get(line[0]+line[3],0) + 1
                #商店发该折扣的券
                b_dis[line[1]+line[3]] = b_dis.get(line[1]+line[3],0) + 1
                #该折扣的该券
                c_dis[line[2]+line[3]] = c_dis.get(line[2]+line[3],0) + 1
                #该折扣的券
                dis[line[3]] = dis.get(line[3],0) + 1
                #该人该券该折扣
                user_c_dis[line[0]+line[2]+line[3]] = user_c_dis.get(line[0]+line[2]+line[3],0) + 1
                #该人该店该折扣
                user_b_dis[line[0]+line[1]+line[3]] = user_b_dis.get(line[0]+line[1]+line[3],0) + 1
                #该店该券该折扣
                b_c_dis[line[1]+line[2]+line[3]] = b_c_dis.get(line[1]+line[2]+line[3],0) + 1
                

    otherF1 = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[5] != 'null':
                otherF1.append([user_c_get_num.get(line[0]+line[2],0),\
                c_num.get(line[2], 0)\
                ,b_num.get(line[1],0)\
                ,user_get_num.get(line[0],0)\
                ,b_c_num.get(line[1]+line[2],0)\
                ,user_b.get(line[0]+line[1],0)\
                ,user_dis.get(line[0]+line[3],0)\
                ,b_dis.get(line[1]+line[3],0)\
                ,c_dis.get(line[2]+line[3],0)\
                ,dis.get(line[3],0)\
                ,user_c_dis.get(line[0]+line[2]+line[3],0)\
                ,user_b_dis.get(line[0]+line[1]+line[3],0)\
                ,b_c_dis.get(line[1]+line[2]+line[3],0)])
    return otherF1

##一个字段统计信息
def one_condition(filename1,filename2):
    offUserUNum={}; offBUNum = {}; offCUNum = {}; offDisUNum = {}; offDUNum = {}
    offUserall = {}; offBall = {}; offCall = {}; offDisall = {}; offUser_u_all = {}
    offUser_no_num = {}; offB_no_num = {};offUser_yes_no_all = {}; offB_yes_no_all = {}
    offB_u_all = {}; offD_yes_no_all = {}; offDis_u_all = {}; offD_no_num = {}; offC_u_all = {}
    d_use_c_15={};user_percent={};user_yes_no_percent={}; user_yes_percent = {}; b_yes_no_percent={}
    offD_u_all = {}; b_has = {}; b_has_yes_no = {}; c_used_percent = {}; dis_used_15 = {}; c_15_percent = {}
    c_consume_time = {}
    user_consume_time = {}
    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                if line[-1] != 'null' and (Date.strptime(line[-1], '%Y%m%d')-Date.strptime(line[-2], '%Y%m%d')).days<15:
                    #有券时该人15天内消费次数                    
                    offUserUNum[line[0]] = offUserUNum.get(line[0], 0) + 1
                    #有券时15天内来该店消费人次
                    offBUNum[line[1]] = offBUNum.get(line[1], 0) + 1
                    #该券15天内被消费次数
                    offCUNum[line[2]] = offCUNum.get(line[2], 0) + 1
                    #该折扣下券15天内被消费次数
                    offDisUNum[line[3]] = offDisUNum.get(line[3], 0) + 1
                    #该距离下券15天内被消费次数
                    offDUNum[line[4]] = offDUNum.get(line[4], 0) + 1
                    #该用户最后一次消费时间
                    timestamp_new = time.mktime(time.strptime(line[-1],'%Y%m%d'))
                    timestamp_old = time.mktime(time.strptime(user_consume_time.get(line[0], '20160101'),'%Y%m%d'))
                    if timestamp_new >= timestamp_old:
                        user_consume_time[line[0]] = line[-1]
                    #该券最后一次被消费时间
                    timestamp_old_c = time.mktime(time.strptime(c_consume_time.get(line[2],'20160101'),'%Y%m%d'))
                    if timestamp_new >= timestamp_old_c:
                        c_consume_time[line[2]] = line[-1]
                #该人领券次数
                offUserall[line[0]] = offUserall.get(line[0],0)+1
                #该商户发券次数
                offBall[line[1]] = offBall.get(line[1],0) + 1
                #该id券发了多少次数
                offCall[line[2]] = offCall.get(line[2],0) + 1
                #该折扣的券有多少张
                offDisall[line[3]] = offDisall.get(line[3],0) + 1
                if line[-1] != 'null':
                    # 有券时该人消费总次数
                    offUser_u_all[line[0]] = offUser_u_all.get(line[0],0) + 1
                    #有券时来该店消费总人次
                    offB_u_all[line[1]] = offB_u_all.get(line[1],0) + 1
                    #该券被消费次数
                    offC_u_all[line[2]] = offC_u_all.get(line[2],0)+ 1
                    #该折扣下券被消费次数
                    offDis_u_all[line[3]] = offDis_u_all.get(line[3], 0) + 1
                    #该距离下券被消费次数
                    
                    offD_u_all[line[4]] = offD_u_all.get(line[4],0) + 1
                    
            if line[2] == 'null' and line[-1] != 'null':
                #没有券时用户消费次数
                offUser_no_num[line[0]] = offUser_no_num.get(line[0],0) + 1
                #没有券时来该店消费人次
                offB_no_num[line[1]] = offB_no_num.get(line[1],0) + 1
                #没有券时该距离下消费券的次数
                offD_no_num[line[4]] = offD_no_num.get(line[4], 0) + 1
            if line[-1] != 'null':
                #不管是否有券该人都消费的总次数
                offUser_yes_no_all[line[0]] = offUser_yes_no_all.get(line[0],0) + 1
                #不管是否有券都来该店消费的总人次
                offB_yes_no_all[line[1]] = offB_yes_no_all.get(line[1],0) + 1
                #不管是否有券该距离下都消费的次数
                offD_yes_no_all[line[4]] = offD_yes_no_all.get(line[4],0) + 1
                
    #15天内使用券占领券次数比例
    for x in offUserall:
        user_percent[x] = offUserUNum.get(line[0], 0)/offUserall[x]
    #有券时15天内消费次数占是否有券都消费次数比例
    for x in offUser_yes_no_all:
        user_yes_no_percent[x] = offUserUNum.get(line[0], 0)/offUser_yes_no_all[x]
    #15天内消费占有券消费比例
    for x in offUser_u_all:
        user_yes_percent[x] = offUserUNum.get(line[0], 0)/offUser_u_all[x]
    #来该店消费次数中有多少比例是有券消费
    for x in offB_yes_no_all:
        b_yes_no_percent[x] = offB_u_all.get(line[1],0)/offB_yes_no_all[x]
    #来该店有券消费次数中有多少比例在15天之内
    for x in offB_u_all:
        b_has[x] = offBUNum.get(line[1], 0)/offB_u_all[x]
    #15天之内有券消费占总消费比例
    for x in offB_yes_no_all:
        b_has_yes_no[x] = offBUNum.get(line[1], 0)/offB_yes_no_all[x]
    #该券15天内被使用的比例
    for x in offCall:
        c_15_percent[x] = offCUNum.get(line[2], 0)/offCall[x]
    #该券被使用的比例
    for x in offCall:
        c_used_percent[x] = offC_u_all.get(line[2],0)/offCall[x]
    #该折扣的券15天内被消费的次数占总消费次数比例
    for x in offC_u_all:
        dis_used_15[x] = offDisUNum.get(line[3], 0)/offC_u_all[x]
    #该距离下15天内的消费记录占有券消费的比例
    for x in offD_yes_no_all:
        d_use_c_15[x] = offDUNum.get(line[4], 0)/offD_yes_no_all[x]
    

    otherF1 = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                #领券时间和最后一次消费时间差
                dateDiff = (Date.strptime(line[5],'%Y%m%d')-Date.strptime(user_consume_time.get(line[0],'20150101'),'%Y%m%d')).days
                dateDiffC = (Date.strptime(line[5],'%Y%m%d')-Date.strptime(c_consume_time.get(line[2],'20150101'),'%Y%m%d')).days
                #feature1
                otherF1.append([dateDiff\
                ,offUserUNum.get(line[0], 0)\
                ,dateDiffC\
                ,offBUNum.get(line[1], 0)\
                ,offCUNum.get(line[2],0)\
                ,offDisUNum.get(line[3], 0)\
                ,offUserall.get(line[0],0)\
                ,offBall.get(line[1],0)\
                ,offCall.get(line[2],0)\
                ,offDisall.get(line[3],0)\
                ,offUser_no_num.get(line[0],0)\
                ,offB_no_num.get(line[1],0)\
                ,offUser_yes_no_all.get(line[0],0)\
                ,offB_yes_no_all.get(line[1],0)\
                ,offDUNum.get(line[4], 0)\
                ,offUser_u_all.get(line[0],0)\
                ,offB_u_all.get(line[1],0)\
                ,offC_u_all.get(line[2],0)\
                ,offDis_u_all.get(line[3], 0)\
                ,offD_no_num.get(line[4], 0)\
                ,offD_yes_no_all.get(line[4],0)\
                ,offD_u_all.get(line[4],0)\
                ,d_use_c_15.get(line[4],0)\
                ,dis_used_15.get(line[3],0)\
                ,c_used_percent.get(line[2],0)\
                ,c_15_percent.get(line[2],0)\
                ,b_has_yes_no.get(line[1],0)\
                ,b_has.get(line[1],0)\
                ,b_yes_no_percent.get(line[1],0)\
                ,user_yes_percent.get(line[0],0)\
                ,offUserall.get(line[0],0)-offUser_u_all.get(line[0],0)\
                ,offCall.get(line[2],0)-offC_u_all.get(line[2],0)\
                ,user_yes_no_percent.get(line[0],0)\
                ,user_percent.get(line[0],0)])       #feature31
    return otherF1

##两个字段联合信息  
def two_condition(filename1,filename2):
    user_b = {}; user_c = {}; user_dis = {}; user_d = {}; b_d = {}; c_d = {}; dis_d = {}
    user_b_all = {}; user_dis_u = {}; user_d_u = {}; user_d_all = {}; b_d_all = {};
    user_b_u = {}; user_c_u = {}; b_c_u = {}; c_dis_u = {}; c_d_u = {};user_b_yes_no={};b_d_yes_no={}
    user_b_u_no = {}; user_c_u_no={}; user_dis_u_no={}
    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null': 
                if line[-1] != 'null' and (Date.strptime(line[-1], '%Y%m%d')-Date.strptime(line[-2], '%Y%m%d')).days<15:
                    #15天内该人在该店里使用券消费的次数                    
                    user_b[line[0]+line[1]] = user_b.get(line[0]+line[1], 0) + 1
                    #15天内该人用该券的次数
                    user_c[line[0]+line[2]] = user_c.get(line[0]+line[2], 0) + 1
                    #15天内该折扣下该人使用券的次数
                    user_dis[line[0]+line[3]] = user_dis.get(line[0]+line[3], 0) + 1
                    #15天内该用户在该距离的时候消费的次数
                   
                    user_d[line[0]+line[4]] = user_d.get(line[0]+line[4], 0) + 1
                    
                    #15天内距离该商户某距离时去商户消费的次数
                    
                    b_d[line[1]+line[4]] = b_d.get(line[1]+line[4], 0) + 1
                    
                    #15天内该距离下该券被使用的次数
                    
                    c_d[line[2]+line[4]] = c_d.get(line[2]+line[4], 0) + 1
                    #15天内该折扣该距离券被使用次数
                    
                    dis_d[line[3]+line[4]] = dis_d.get(line[3]+line[4], 0) + 1
                
                if line[-1] != 'null':
                   #有券时该人来该店使用券的次数
                   user_b_u[line[0]+line[1]] = user_b_u.get(line[0]+line[1],0) + 1
                   #该人使用该券的次数
                   user_c_u[line[0]+line[2]] = user_c_u.get(line[0]+line[2],0) + 1
                   #该人使用该折扣券的次数
                   user_dis_u[line[0]+line[3]] = user_dis_u.get(line[0]+line[3],0) + 1
                   #该人在该距离时去消费的次数
                   
                   user_d_u[line[0]+line[4]] = user_d_u.get(line[0]+line[4],0) + 1
                   #该商店的该券被使用次数
                   b_c_u[line[1]+line[2]] = b_c_u.get(line[1]+line[2],0) + 1
                   #该券该折扣时被使用次数
                   c_dis_u[line[2]+line[3]] = c_dis_u.get(line[2]+line[3],0) + 1
                   #该券该距离时被使用次数
                   c_d_u[line[2]+line[4]] = c_d_u.get(line[2]+line[4], 0) + 1
                if line[-1] == 'null':
                   #有券时该人来该店使用券的次数
                   user_b_u_no[line[0]+line[1]] = user_b_u_no.get(line[0]+line[1],0) + 1
                   #该人使用该券的次数
                   user_c_u_no[line[0]+line[2]] = user_c_u_no.get(line[0]+line[2],0) + 1
                   #该人使用该折扣券的次数
                   user_dis_u_no[line[0]+line[3]] = user_dis_u_no.get(line[0]+line[3],0) + 1
                   
                 
            if line[2] == 'null' and line[-1] != 'null':           
                #没有券时该用户在该商店消费次数
                user_b_all[line[0]+line[1]] = user_b_all.get(line[0]+line[1], 0) + 1
                #该用户在该距离下消费次数
                
                user_d_all[line[0]+line[4]] = user_d_all.get(line[0]+line[4], 0) + 1  
                #该商店在该距离下去消费的总人次
                
                b_d_all[line[1]+line[4]] = b_d_all.get(line[1]+line[4],0) + 1
            if line[-1] != 'null':
                #不管有没有券该人都来该商户消费次数
                user_b_yes_no[line[0]+line[1]] = user_b_yes_no.get(line[0]+line[1],0) + 1
                #不管有没有券该距离该商户都有人来消费券的次数
                
                b_d_yes_no[line[1]+line[4]] = b_d_yes_no.get(line[1]+line[4],0) + 1
    
    otherF2 = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                #feature32
               otherF2.append([user_b.get(line[0]+line[1], 0)\
               ,user_c.get(line[0]+line[2], 0)\
               ,user_dis.get(line[0]+line[3], 0)\
               ,user_d.get(line[0]+line[4], 0)\
               ,b_d.get(line[1]+line[4], 0)\
               ,c_d.get(line[2]+line[4], 0)\
               ,dis_d.get(line[3]+line[4], 0)\
               ,user_b_u.get(line[0]+line[1],0)
               ,user_c_u.get(line[0]+line[2],0)\
               ,user_dis_u.get(line[0]+line[3],0)\
               ,user_d_u.get(line[0]+line[4],0)\
               ,c_dis_u.get(line[2]+line[3],0)\
               ,c_d_u.get(line[2]+line[4], 0)\
               ,user_b_all.get(line[0]+line[1], 0)\
               ,user_d_all.get(line[0]+line[4], 0)\
               ,b_d_all.get(line[1]+line[4],0)\
               ,user_b_yes_no.get(line[0]+line[1],0)\
               ,b_d_yes_no.get(line[1]+line[4],0)\
               ,user_b_u_no.get(line[0]+line[1],0)\
               ,user_c_u_no.get(line[0]+line[2],0)\
               ,user_dis_u_no.get(line[0]+line[3],0)])        #feature49
    return otherF2

##三个字段的联合信息
def three_condition(filename1,filename2):
    user_b_c = {}; user_b_dis = {};user_c_dis = {}; user_c_d = {}; user_dis_d = {}; b_c_dis = {}; b_c_d = {}; b_dis_d = {}; c_dis_d = {}
    user_b_c_all = {}; user_b_dis_all = {}; user_c_dis_all = {}; user_c_d_all = {}; user_dis_d_all = {}; b_c_dis_all = {}; b_c_d_all = {}; b_dis_d_all = {}; c_dis_d_all = {}
    user_b_c_all_no={}
    
    
    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                if line[-1] != 'null' and (Date.strptime(line[-1], '%Y%m%d')-Date.strptime(line[-2], '%Y%m%d')).days<15:
                    #15天内该人在该商户使用消费该券的次数                    
                    user_b_c[line[0]+line[1]+line[2]] = user_b_c.get(line[0]+line[1]+line[2], 0) + 1
                    #15天内该人在该商店该折扣下使用券的次数
                    user_b_dis[line[0]+line[1]+line[3]] = user_b_dis.get(line[0]+line[1]+line[3], 0) + 1
                    #15天内该人消费该折扣的该券的次数
                    user_c_dis[line[0]+line[2]+line[3]] = user_c_dis.get(line[0]+line[2]+line[3], 0) + 1
                    #15天内该距离下该用户使用该券的次数
                    
                    user_c_d[line[0]+line[2]+line[4]] = user_c_d.get(line[0]+line[2]+line[4], 0) + 1
                    #15天内在该距离该折扣下该人使用券的次数
                    
                    user_dis_d[line[0]+line[3]+line[4]] = user_dis_d.get(line[0]+line[3]+line[4], 0) + 1
                    #15天内该商户的该折扣的该券被使用的人次
                    b_c_dis[line[1]+line[2]+line[3]] = b_c_dis.get(line[1]+line[2]+line[3], 0) + 1
                    #15天内该距离该商户使用该券的人次
                    
                    b_c_d[line[1]+line[2]+line[4]] = b_c_d.get(line[1]+line[2]+line[4], 0) + 1
                    #15天内该距离该商户该折扣来店里消费的次数
                    
                    b_dis_d[line[1]+line[3]+line[4]] = b_dis_d.get(line[1]+line[3]+line[4], 0) + 1
                    #15天内该距离该折扣的该券被使用次数
                    
                    c_dis_d[line[2]+line[3]+line[4]] = c_dis_d.get(line[2]+line[3]+line[4], 0) + 1
                if line[-1] != 'null':
                    #该人在该商户使用消费该券的次数
                    user_b_c_all[line[0]+line[1]+line[2]] = user_b_c_all.get(line[0]+line[1]+line[2], 0) + 1
                    #该人在该商店该折扣下使用券的次数
                    user_b_dis_all[line[0]+line[1]+line[3]] = user_b_dis_all.get(line[0]+line[1]+line[3], 0) + 1
                    #该人消费该折扣的该券的次数
                    user_c_dis_all[line[0]+line[2]+line[3]] = user_c_dis_all.get(line[0]+line[2]+line[3], 0) + 1
                    #该距离下该用户使用该券的次数
                    
                    user_c_d_all[line[0]+line[2]+line[4]] = user_c_d_all.get(line[0]+line[2]+line[4], 0) + 1
                    #在该距离该折扣下该人使用券的次数
                    
                    user_dis_d_all[line[0]+line[3]+line[4]] = user_dis_d_all.get(line[0]+line[3]+line[4], 0) + 1
                    #该商户的该折扣的该券被使用的人次
                    b_c_dis_all[line[1]+line[2]+line[3]] = b_c_dis_all.get(line[1]+line[2]+line[3], 0) + 1
                    #该距离该商户使用该券的人次
                    
                    b_c_d_all[line[1]+line[2]+line[4]] = b_c_d_all.get(line[1]+line[2]+line[4], 0) + 1
                    #该距离该商户该折扣来店里消费的次数
                    
                    b_dis_d_all[line[1]+line[3]+line[4]] = b_dis_d_all.get(line[1]+line[3]+line[4], 0) + 1
                    #该距离该折扣的该券被使用次数
                    
                    c_dis_d_all[line[2]+line[3]+line[4]] = c_dis_d_all.get(line[2]+line[3]+line[4], 0) + 1
                if line[-1] == 'null':
                    #该人在该商户bu使用消费该券的次数
                    user_b_c_all_no[line[0]+line[1]+line[2]] = user_b_c_all_no.get(line[0]+line[1]+line[2], 0) + 1
                    
                
    otherF3 = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                #feature50
                otherF3.append([user_b_c.get(line[0]+line[1]+line[2], 0)\
                ,user_b_dis.get(line[0]+line[1]+line[3], 0)\
                ,user_c_dis.get(line[0]+line[2]+line[3], 0)\
                ,b_c_dis.get(line[1]+line[2]+line[3], 0)\
                ,user_c_d.get(line[0]+line[2]+line[4], 0)\
                , user_dis_d.get(line[0]+line[3]+line[4], 0)\
                ,b_c_d.get(line[1]+line[2]+line[4], 0)\
                , b_dis_d.get(line[1]+line[3]+line[4], 0)\
                ,c_dis_d.get(line[2]+line[3]+line[4], 0)\
                ,user_b_c_all.get(line[0]+line[1]+line[2], 0)\
                ,user_b_dis_all.get(line[0]+line[1]+line[3], 0)\
                ,user_c_dis_all.get(line[0]+line[2]+line[3], 0)\
                ,user_c_d_all.get(line[0]+line[2]+line[4], 0)\
                ,user_dis_d_all.get(line[0]+line[3]+line[4], 0)\
                ,b_c_dis_all.get(line[1]+line[2]+line[3], 0)\
                ,b_c_d_all.get(line[1]+line[2]+line[4], 0)\
                ,b_dis_d_all.get(line[1]+line[3]+line[4], 0)\
                ,c_dis_d_all.get(line[2]+line[3]+line[4], 0)\
                ,user_b_c_all_no.get(line[0]+line[1]+line[2], 0)])#       #feature67
    return otherF3

##四个字段联合信息（现已废除） 
def four_condition(filename1,filename2):
    user_b_c_dis = {}; b_c_dis_d = {}; user_c_dis_d = {}; 
    user_b_c_dis_all = {}; b_c_dis_d_all = {}; user_c_dis_d_all = {}
    

    with open(filename1) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                if line[-1] != 'null' and (Date.strptime(line[-1], '%Y%m%d')-Date.strptime(line[-2], '%Y%m%d')).days<15:
                    #15天内该用户在该商店的该折扣的该券使用次数
                    user_b_c_dis[line[0]+line[1]+line[2]+line[3]] = user_b_c_dis.get(line[0]+line[1]+line[2]+line[3], 0) + 1
                    #15天内该商店该距离该折扣的该券使用次数
                    
                    b_c_dis_d[line[1]+line[2]+line[3]+line[4]] = b_c_dis_d.get(line[1]+line[2]+line[3]+line[4], 0) + 1
                    #15天内该人在该距离该折扣的该券使用次数
                    
                    user_c_dis_d[line[0]+line[2]+line[3]+line[4]] = user_c_dis_d.get(line[0]+line[2]+line[3]+line[4], 0) + 1
                    
                if line[-1] != 'null':
                    #该用户在该商店的该折扣的该券使用次数
                    user_b_c_dis_all[line[0]+line[1]+line[2]+line[3]] = user_b_c_dis_all.get(line[0]+line[1]+line[2]+line[3], 0) + 1
                    #该商店该距离该折扣的该券使用次数
                    
                    b_c_dis_d_all[line[1]+line[2]+line[3]+line[4]] = b_c_dis_d_all.get(line[1]+line[2]+line[3]+line[4], 0) + 1
                    #该人在该距离该折扣的该券使用次数
                    
                    user_c_dis_d_all[line[0]+line[2]+line[3]+line[4]] = user_c_dis_d_all.get(line[0]+line[2]+line[3]+line[4], 0) + 1
                    
    otherF4 = []
    with open(filename2) as fs:
        data = csv.reader(fs)
        for line in data:
            if line[2] != 'null':
                #feature68
                otherF4.append([])#user_b_c_dis.get(line[0]+line[1]+line[2]+line[3], 0)\
                #,b_c_dis_d.get(line[1]+line[2]+line[3]+line[4], 0)\
                #user_c_dis_d.get(line[0]+line[2]+line[3]+line[4], 0)\
                #,user_b_c_dis_all.get(line[0]+line[1]+line[2]+line[3], 0)\
                #,b_c_dis_d_all.get(line[1]+line[2]+line[3]+line[4], 0)\
                #,user_c_dis_d_all.get(line[0]+line[2]+line[3]+line[4], 0)])#feature73
    return otherF4

#总的样本数据  
def getData(otherF1, otherF2,otherF3,otherF,otherFonline):
    data = []
    m = len(otherF1)
    for i in xrange(m):
        data.append(otherF1[i]+otherF2[i]+otherF3[i]+otherF[i]+otherFonline[i])
    return data
    
#对距离进行独热编码,距离特征
def distanceFtrain1(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat1 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[4],0)
                    dFeat1.append(b)
                else:
                    dFeat1.append(b)
    return dFeat1
    
def distanceFtrain2(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat2 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[4],0)
                    dFeat2.append(b)
                else:
                    dFeat2.append(b)
    return dFeat2
def distanceFtrain3(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat3 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[1]+line[4],0)
                    dFeat3.append(b)
                else:
                    dFeat3.append(b)
    return dFeat3

def distanceFtrain4(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat4 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[2]+line[4],0)
                    dFeat4.append(b)
                else:
                    dFeat4.append(b)
    return dFeat4

def distanceFtrain5(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat5 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[3]+line[4],0)
                    dFeat5.append(b)
                else:
                    dFeat5.append(b)
    return dFeat5
    
def distanceFtrain7(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat7 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[2]+line[4],0)
                    dFeat7.append(b)
                else:
                    dFeat7.append(b)
    return dFeat7
    
def distanceFtrain8(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat8 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[3]+line[4],0)
                    dFeat8.append(b)
                else:
                    dFeat8.append(b)
    return dFeat8

def distanceFtrain9(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat9 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[1]+line[2]+line[4],0)
                    dFeat9.append(b)
                else:
                    dFeat9.append(b)
    return dFeat9

def distanceFtrain10(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat10 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[1]+line[3]+line[4],0)
                    dFeat10.append(b)
                else:
                    dFeat10.append(b)
    return dFeat10

def distanceFtrain11(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat11 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[2]+line[3]+line[4],0)
                    dFeat11.append(b)
                else:
                    dFeat11.append(b)
    return dFeat11
    
def distanceFtrain12(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat12 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[1]+line[2]+line[3]+line[4],0)
                    dFeat12.append(b)
                else:
                    dFeat12.append(b)
    return dFeat12
    
def distanceFtrain13(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat13 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[2]+line[3]+line[4],0)
                    dFeat13.append(b)
                else:
                    dFeat13.append(b)
    return dFeat13



"""
def distanceFtrain6(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat6 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[1]+line[4],0)
                    dFeat6.append(b)
                else:
                    dFeat6.append(b)
    return dFeat6    
    
def distanceFtrain14(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat14 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[1]+line[3]+line[4],0)
                    dFeat14.append(b)
                else:
                    dFeat14.append(b)
    return dFeat14
    
def distanceFtrain15(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat15 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[1]+line[2]+line[4],0)
                    dFeat15.append(b)
                else:
                    dFeat15.append(b)
    return dFeat15
    
def distanceFtrain16(distanceDict):
    dList = ['0','1','2','3','4','5','6','7','8','9','10']
    m = len(dList); dFeat16 = []
    with open("splitData\data2.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            b = [0]*m
            if line[2] != 'null':
                if (line[4] != 'null'):
                    b[dList.index(line[4])] = distanceDict.get(line[0]+line[1]+line[2]+line[3]+line[4],0)
                    dFeat16.append(b)
                else:
                    dFeat16.append(b)
    return dFeat16
"""    
    
#对折扣优惠进行独热编码，优惠特征
def discountFtrain(offDisProb):
   disList = [x for x in offDisProb]
   m = len(disList); disFeat = []
   with open("ccf_offline_stage1_test_revised.csv") as fs:
       data = csv.reader(fs)
       for line in data:
           b = [0]*m
           if line[2] != 'null':
               if (line[3] != 'null') and (line[3] in offDisProb):
                   b[disList.index(line[3])] = offDisProb.get(line[3],0)
                   disFeat.append(b)
               else:
                   disFeat.append(b)
   return disFeat
#时间独热编码 时间特征



    
#从table1数据直接统计用户使用券的概率  and 券被使用的概率
def offDirectProb():
    offData = []; 
    offUserNum={}; #某id用户个数
    offUseCouponNum = {}; #某用户使用的券个数
    offCNum = {}; #某id券个数
    offCUseNum={}#使用的id券个数       
    offBNum = {}#某商户个数
    offBUNum = {}#某商户对应的券使用个数
    offDNum = {}#某距离个数
    offDUNum = {}#某距离对应的券使用个数
    offDisNum = {}#某折扣个数
    offDisUNum = {}#某折扣对用的券使用个数
    offUserProb = {}; #用户概率  权重5
    offCProb = {}#券概率       权重2
    offBProb = {}#商家概率   权重2
    offDProb = {}#距离概率   权重4
    offDisProb = {}#折扣概率  权重3
    with open("ccf_offline_stage1_train.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            offData.append(line) 
    m = len(offData)
    #统计各种总数
    for i in xrange(m):
        if offData[i][2] != 'null': #判断是否领券
            offUserNum[offData[i][0]] = offUserNum.get(offData[i][0],0) + 1
            offBNum[offData[i][1]] = offBNum.get(offData[i][1],0) + 1
            offCNum[offData[i][2]] = offCNum.get(offData[i][2],0) + 1
            if offData[i][4] != 'null':#在领券的情况下距离统计
                offDNum[offData[i][4]] = offDNum.get(offData[i][4],0) + 1
        if offData[i][3] != 'null':#不为null必定领了券
            offDisNum[offData[i][3]] = offDisNum.get(offData[i][3],0) + 1
        
    #统计各种使用个数
    for i in xrange(m):
        if (offData[i][2] != 'null') and (offData[i][-1] != 'null') and (Date.strptime(offData[i][-1], '%Y%m%d')-Date.strptime(offData[i][-2], '%Y%m%d')).days<15:#判断是否使用券
            offUseCouponNum[offData[i][0]] = offUseCouponNum.get(offData[i][0],0) + 1
            offBUNum[offData[i][1]] = offBUNum.get(offData[i][1],0) + 1 
            offCUseNum[offData[i][2]] = offCUseNum.get(offData[i][2],0) + 1 
            if offData[i][3] != 'null':
                offDisUNum[offData[i][3]] = offDisUNum.get(offData[i][3],0) + 1 
            if offData[i][4] != 'null':
                offDUNum[offData[i][4]] = offDUNum.get(offData[i][4],0) + 1
            
            
    for user in offUserNum:
        if offUseCouponNum.has_key(user):
            offUserProb[user] = offUserProb.get(user,0) + offUseCouponNum[user]/offUserNum[user]
        else:
            offUserProb[user] = 0.0
    
    for user in offBNum:
        if offBUNum.has_key(user):
            offBProb[user] = offBProb.get(user,0) + offBUNum[user]/offBNum[user]
        else:
            offBProb[user] = 0.0    
    
    for user in offCNum:
        if offCUseNum.has_key(user):
            offCProb[user] = offCProb.get(user,0) + offCUseNum[user]/offCNum[user]
        else:
            offCProb[user] = 0.0
    for user in offDisNum:
        if offDisUNum.has_key(user):
            offDisProb[user] = offDisProb.get(user,0) + offDisUNum[user]/offDisNum[user]
        else:
            offDisProb[user] = 0.0
            
    for user in offDNum:
        if offDUNum.has_key(user):
            offDProb[user] = offDProb.get(user,0) + offDUNum[user]/offDNum[user]
        else:
            offDProb[user] = 0.0
            
            
    return offDisProb#offUserProb, offBProb, offCProb, offDisProb,offDProb
    #return offUseCouponNum, offBUNum, offCUseNum, offDisUNum, offDUNum


#统计特征概率
def predictProb(offTestData, offUserProb, offBProb, offCProb, offDisProb, offDProb):
    prob = []
    for line in offTestData:
        p = 0.5*offUserProb.get(line[0],0) + 0.05*offBProb.get(line[1],0) + \
        0.05*offCProb.get(line[2],0) + 0.2*offDisProb.get(line[3],0) + 0.2*offDProb.get(line[4],0)
        prob.append(p)
    return prob

        
#保存数据
def storeData(trainData,testData,label):
        
    with open('trainData_all_month.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in trainData:
            myWrite.writerow(line)
    
    with open('testData_all_month.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        for line in testData:
            myWrite.writerow(line)
    
    with open('label_all.csv','wb+') as fs:
        myWrite = csv.writer(fs)
        myWrite.writerow(label)
    

            
            
#统计训练样本中的用户
def userNum():
    user_num = {}
    with open("ccf_offline_stage1_train.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            user_num[line[0]] = user_num.get(line[0], 0) + 1
    return user_num
        

#测试样本里的用户是否都在训练样本中
def yes_or_no(offTestData, user_num):
    result = []
    for line in offTestData:
        if(user_num.has_key(line[0]) == 0):
            result.append(line[0])
    return result


            
def importantFeat(lr,trainData,testData):
    index = []; a=[]; b=[]
    feature_important = list(lr.feature_importances_)
    index = [i for i in xrange(len(feature_important)) if feature_important[i]==0]
    for line in trainData:
        for i in index:
            line[i] = -1
    for line in testData:
        for i in index:
            line[i] = -1
    trainData = np.array(trainData); testData = np.array(testData)
    for line in trainData:
        line = list(line[line>=0])
        a.append(line)
    for line in testData:
        line = list(line[line>=0])
        b.append(line)
    return a,b
    
def merge(trainData2,trainData3,trainData4,trainData5,trainData6):#,label2,label3,label4,label5,label6):
    trainData = []; 
    #label = label2+label3+label4+label5+label6
    for line in trainData2:
        trainData.append(line)
    for line in trainData3:
        trainData.append(line)
    for line in trainData4:
        trainData.append(line)
    for line in trainData5:
        trainData.append(line)
    for line in trainData6:
        trainData.append(line)
    return trainData#,label
#合并测试数据    
def mergetest(testData_off,testData_online,trainData_off,trainData_online):
    testData=[];trainData = []
    for line in testData_off:
        testData.append(line)
    for line in testData_online:
        testData.append(line)
    for line in trainData_off:
        trainData.append(line)
    for line in trainData_online:
        trainData.append(line)
    return testData
            

    
    
        
        
        
        
        
        
        
