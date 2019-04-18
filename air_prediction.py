# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle
import numpy as np
import math
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from sklearn.preprocessing import MinMaxScaler   #这是标准化处理的语句，很方便，里面有标准化和反标准化。。
np.random.seed(1337)  # for reproducibility
# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH  #配置的环境
T = 24  # number of time intervals in one day   一天的周期迭代次数

lr = 0.0002  # learning rate
len_closeness = 12  # length of closeness dependent sequence   考虑的相邻的迭代次数
len_period = 4  # length of peroid dependent sequence       以相邻周期四个作为预测趋势
len_trend = 4  # length of trend dependent sequence    以前面4个作为趋势性
nb_residual_unit = 8   # number of residual units   残差单元数量

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days    使用10天数据进行测试
days_test = 10
len_test = T * days_test   #测试用的时间戳数量
map_height, map_width = 35, 11  # grid size   每个代表流量意义的格点图的大小为16*8
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81  # 共有81个基于网格点的区域， 每个区域至少有1个自行车站
# m_factor 计算得到影响因素   影响因子的具体计算为什么这样算
path_result = 'RET'
path_model = 'MODEL'

'''
下面的程序将执行预测人物，  我打算把 大的数据集加载过来，找到其中

'''
def build_model(external_dim):
    #创建模型时   首先指定进行组合时的参数    将配置分别放进不同的区域中（就是相近性的长度 ，周期性的长度等）
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    '''
    趋势性的数据暂时不要
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None
    '''
    #根据不同的配置定义残差神经网络模型  这个stresnet是定义好的，传入关于不同方面的配置，最后会返回根据参数组合好的模型
    model = stresnet(c_conf=c_conf, p_conf=p_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr) #接下来 定义学习率和损失函数值
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    #model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model

def main():
    # 加载预测需要用到的数据
    print("loading data...")
    X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True)

    for _X in X_train:
        print('theshape  ',_X.shape, )

    model = build_model(9)
    fname_param='/home/fly/PycharmProjects/DeepST-KDD/scripts/AirPrediction/MODEL/c12.p4.t4.resunit8.lr0.0002.cont.best.h5'
    model.load_weights(fname_param)
    #开始使用模型进行预测    这里有毒，草，   这里需要的输入是三个输入
    print('Y_train[0]:',type(Y_train))

    '''
    这里的问题困惑了我很长时间，也怪我对kreas和 神经网络的训练机制不够理解的通透，
    第一点：这里 因为自己创建的模型的输入是有批次数的，所以对应进行预测的时候 输入的数据也得有批次数（kreas的预测、评估、训练都是按批次的），对于predict预测也是这样的，  所以这里我是限定了维度，包装成了需要的
    输入的形式，这是根据X的形式包装的一个特征数据对应的输出预测。
    第二点: 在于对批次的理解上，和师姐交流了下，对于 图像数据是4唯的网络输入（批次数，通道，长度，宽度），对于lstm是3维的输入（批次数，记忆体长度..），其中一次训练时候
    是按批次的，相当于加了个维度，按正常的方式由前往后跑，一个批次每个数据都会得到一个损失值，得到的损失值进行批次内sum合并之后再进行一次传播，所以理明白 模型的训练是以
    一个批次的数据为单位进行训练的，  跑模型时候批次的作用体现在扩展维度上，  批次内数据也就是一个高唯独内的数据。
    
    还有构造的模型中根本没有考虑批次的事情，将批次的融入是kreas中的fit所作的事情， 可以看到模型中根本就没有batch_size的事情。
    第三点：kreas中模型的输入不像TF是按名 feed传值的，kreas中的方式是  按照计算流图找到开始的传入将数据传给神经网络，对应放到输入的位置，自己传入所有数据，在fit中指定
    好批次就好，数据会按批次训练。。。    
    其实这里有个问题是外部因素的融入我只融入了一天一个唯独的节假日影响（是以后一个时刻点所对应的数据），
    
    '''
    #========================      进行预测      ===========================#
    #  下面是一种模式，已经掌握了其思想
    y_pred=model.predict([X_test[0][0].reshape(1,12,35,11),X_test[1][0].reshape(1,4,35,11),X_test[2][0].reshape(1,9)],batch_size=1)
    print('成功得到时刻点',timestamp_train[0],'的一个预测结果：',y_pred.shape)
    the_last_list=[]
    #现在将不同时刻点的数据按列表方式进行组合成一个二维的，准备存到csv文件中去。
    #========================      遍历和组装      ===========================#
    #下面的方式 是一种遍历， array中的特征数据 全放在一个列表中去，   发现直接时间拼接的方式不对,最后也终于攻克了，就是列表的拼接方式。。。
    for i in range(len(y_pred[0])):
        for ii in y_pred[0][i]:
            #这里必须写三步，这是我的新发现。 不这样写 list就变成None了，不可思议爱
            ii_list=ii.tolist()
            ii_list.append(timestamp_train[i])
            the_last_list.append(ii_list)
            #print(ii_list.append(timestamp_train[i][0]))
            #the_last_list.append()

    #print(y_pred[0][0][0].tolist().append('dsa'))
    #print('the_last_list  length:',the_last_list)
    pred_array=pd.DataFrame(the_last_list,columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3','utc_time'])
    pred_array.to_csv('/home/fly/PycharmProjects/DeepST-KDD/data/my_process/the_predict_data.csv',index=False)
    print(pred_array)
    #会得到 2100*12 行的表， 其中最后一行时间数据可以不要 进行反标准化  会在和标准化同一个文件中执行


    #========================      反标准化      ===========================#
    #这是正标准化
    # scaler = MinMaxScaler(feature_range=(-1, 1)) #数值限定到-1到1之间 看出来了，这是在同一个scaler下，将tranform转换后的数据，使用reverse还原回来，scaler有记忆的，所以在这里没用
    # scaler.fit(pred_array) #这是个fit过程， 这个过程会 计算以后用于缩放的平均值和标准差， 记忆下来
    # X_tr = scaler.fit_transform(pred_array)  #使用记忆的数据进行转换
    # #这是反标准化
    # pred_array = scaler.inverse_transform(pred_array)  #使用记忆的数据反转换
    # pred_array=pd.DataFrame(pred_array)
if __name__ == '__main__':
    main()