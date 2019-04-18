# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle
import numpy as np

from deepst.datasets import load_stdata
from deepst.preprocessing import MinMaxNormalization
from deepst.preprocessing import timestamp9vec   #这个加入函数的方法
from deepst.preprocessing import remove_incomplete_days
from deepst.config import Config
from deepst.datasets.STMatrix import STMatrix
from deepst.preprocessing import timestamp2vec
np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH

#加载训练数据，并对数据进行处理
def load_data(T=24, nb_flow=1, len_closeness=None, len_period=None, len_trend=None,len_test=None, preprocess_name='preprocessing.pkl', meta_data=True):
    assert(len_closeness + len_period > 0)
    # load data   加载两个部分，各日期和其所对应的数据[流量和外部因素数据]
    data, timestamps,data_air,pred_48_air = load_stdata('/home/fly/PycharmProjects/DeepST-KDD for_train/data/4-14_new_data/2017_data/final_merge_aq_grid_meo_with_weather4-17.h5')
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    #data, timestamps = remove_incomplete_days(data, timestamps, T)
    #data = data[:, :nb_flow]#只是需要两个图，其实也就两个图数据吧
    #是对预处理函数的应用，现在对数据进行很多预处理设置 去除数据中小于0的数
    #data[data < 0] = 0.
    data_all = [data]
    #获取得到所有时间戳  这都是经过处理过的
    timestamps_all = [timestamps]
    data_air_all=[data_air]
    pred_48_air_all=[pred_48_air]
    #想办法进行替换，测试用 35*3维度的数据

    # minmax_scale
    data_train = data[:-len_test]  #这是一种分离，除了测试尺寸外的 都考虑作为训练集
    #print('train_data shape: ', data_train.shape)
    #mmn = MinMaxNormalization() #对所有要进行训练的数据进行 标准化处理
    #mmn.fit(data_train)   #使用训练数据进行fit（预先的放置最大最小数作用），之后对所有数据进行标准化处理，
    #data_all_mmn = []
    # for d in data_all:
    #     data_all_mmn.append(mmn.transform(d))  #transform函数是对公式的进一步计算，这是对所有数据处理，这就是标准化的最后结果。
    # fpkl = open('preprocessing.pkl', 'wb')
    # for obj in [mmn]:
    #     #这部分将预处理过的结果保存到pkl文件中去，使用dump方式
    #     pickle.dump(obj, fpkl) #pickle.dump(obj, file, [,protocol]) 注解:将对象obj保存到文件file中去 (将预处理过的数据存起来)
    # fpkl.close()
    XC, XP, XT,X48_air = [], [], [],[]
    Y = []
    timestamps_Y = []
    for data, timestamps,data_air,pred_48_air in zip(data_all, timestamps_all,data_air_all,pred_48_air_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps,data_air,pred_48_air ,T, CheckComplete=False)#通过将data和timestamp组合创建时空矩阵
        #对时空矩阵st进行 create_dataset函数分解分别得到_XC、_XP、_XT、_Y四大部分（有时间特性的数据组合）
        _XC, _XP,_XT, _Y, _timestamps_Y= st.create_dataset(len_closeness=len_closeness, len_period=len_period,len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        #X48_air.append(_X48_air)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    #X48_air=np.vstack(X48_air)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,  "Y shape:", Y.shape)
    XC_train, XP_train,XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test,XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    #print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp9vec(timestamps_Y)  #获取长度为9，第8位是工作日标识，9位置是节假日标识
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None


    #------------------  加入 预报天气信息  ---------------#
    #X48_air_train, X48_air_test = X48_air[:-len_test], X48_air[-len_test:]
    #X_train.append(X48_air_train)
    #X_test.append(X48_air_test)
    # --------------------------------------------------#

    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )

   # print('timestamp_train:', timestamp_train.shape)
    print('timestamp_train bbbbb:', timestamp_train)
    print('timestamp_test  bbbbb:', len(timestamp_test))
    #metadata_dim 就是8，外部因素数据早已经存好的
    return X_train, Y_train, X_test, Y_test, metadata_dim, timestamp_train, timestamp_test
