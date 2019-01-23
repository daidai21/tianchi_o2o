# -*- coding:utf-8 -*-


'''
using:绘制散点图，查看多个模型预测的数据走势，确定模型是互补的还是相同的
input:
output:matplotlib 图
'''


# import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# func
def read_csv__drop_nouse(path):
    '''
    using:读取csv文件；只留最后一列数据并转为一维数组
    intput:csv文件路径
    output:csv文件最后一列label属性的一维数组
    '''
    data_df = pd.read_csv(path, names=['user-id', 'coupon-id', 'date', 'rate'])  # read csv
    # print(data.head())
    # print(data.describe())
    data_df = data_df.rate  # select rate column
    # print(data.head())
    # print(data.describe())
    data_np = data_df.values  # df to np
    # print(data_np)
    return data_np


# func draw 3 graph
def draw_graph_3(y_1, y_2, y_3):
    '''
    using:绘制散点图
    input:一维数组
    output:
    '''
    x = [i for i in range(1, y_1.size + 1)]

    # first graph
    plt.subplot(311)
    plt.plot(x, y_1)
    plt.title('0.52054')
    # second graph
    plt.subplot(312)
    plt.plot(x, y_2)
    plt.title('0.53321810')
    # three graph
    plt.subplot(313)
    plt.plot(x, y_3)
    plt.title('0.64030')
    plt.show()


# func draw 4 graph
def draw_graph_4(y_1, y_2, y_3, y_4):
    '''
    using:绘制散点图
    input:一维数组
    output:
    '''
    x = [i for i in range(1, y_1.size + 1)]

    # first graph
    plt.subplot(411)
    plt.plot(x, y_1)
    plt.title('0.78835167')
    # second graph
    plt.subplot(412)
    plt.plot(x, y_2)
    plt.title('0.78525997')
    # three graph
    plt.subplot(413)
    plt.plot(x, y_3)
    plt.title('0.79630296')
    # four graph
    plt.subplot(414)
    plt.plot(x, y_4)
    plt.title('0.79667921')
    plt.show()


# main call
if __name__ == "__main__":
    # 设置路径
    file_csv_1 = '1.csv'
    file_csv_2 = '2.csv'
    # file_csv_3 = '3.csv'
    file_csv_4 = '4.csv'
    file_csv_5 = '5.csv'
    file_csv_6 = '6.csv'
    file_csv_7 = '7.csv'
    file_csv_8 = '8.csv'

    # call read_csv__drop_nouse
    np_1 = read_csv__drop_nouse(file_csv_1)
    np_2 = read_csv__drop_nouse(file_csv_2)
    # np_3 = read_csv__drop_nouse(file_csv_3)
    np_4 = read_csv__drop_nouse(file_csv_4)
    np_5 = read_csv__drop_nouse(file_csv_5)
    np_6 = read_csv__drop_nouse(file_csv_6)
    np_7 = read_csv__drop_nouse(file_csv_7)
    np_8 = read_csv__drop_nouse(file_csv_8)

    print(np_1.size, 
          np_2.size, 
          # np_3.size, 
          np_4.size, 
          np_5.size, 
          np_6.size, 
          np_7.size, 
          np_8.size)

    # draw 1,2,4
    # draw_graph_3(np_1, np_2, np_4)

    # draw 5,6,7,8
    draw_graph_4(np_5, np_6, np_7, np_8)
