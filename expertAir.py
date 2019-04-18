# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptBikeNYC.py
"""
from __future__ import print_function
import os
import pickle
import numpy as np
import tensorflow as tf
#import math
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from  ma_util.show_pkl import showResult
from deepst.datasets import BikeNYC
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

np.random.seed(1337)  # for reproducibility

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH  #配置的环境
nb_epoch = 200  # number of epoch at training stage  训练时的迭代次数
nb_epoch_cont = 200  # number of epoch at training (cont) stage    阶段性的你
batch_size = 16  # batch size   每次训练的批次
T = 24  # number of time intervals in one day   一天的周期迭代次数

lr = 0.0001  # learning rate
len_closeness = 12  # length of closeness dependent sequence   考虑的相邻的迭代次数
len_period = 1  # length of peroid dependent sequence       以相邻周期四个作为预测趋势
len_trend =4   # length of trend dependent sequence    以前面4个作为趋势性
nb_residual_unit =  6  # number of residual units   残差单元数量


nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days    使用10天数据进行测试
days_test = 15
len_test = T * days_test   #测试用的时间戳数量
map_height, map_width = 35, 12  # grid size   每个代表流量意义的格点图的大小为16*8
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81  # 共有81个基于网格点的区域， 每个区域至少有1个自行车站
# m_factor 计算得到影响因素   影响因子的具体计算为什么这样算
path_result = 'RET'
path_model = 'MODEL'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#放入评价标准函数， 突然发现 直接用tf.reduce_mean函数居然可以用，^_^   不懂，把不可以用的mean换成 tensorflow中的mean就行了(报错里面说tensor类型的问题，所以用tensor去做mean就行了)
#现在已经在做线下验证了，  rmse的绝对值就是要预测结果（标准化前后是有差距的）
def smape(actual, predicted):

    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    return 2 * tf.reduce_mean(np.divide(a, b, out=np.zeros_like(a),where=b!= 0,casting='unsafe').tolist())
def mape_ling(actual, predicted):
    epsilon = 0.1
    summ = tf.maximum(tf.abs(actual) + tf.abs(predicted) + epsilon, 0.5 + epsilon)
    smape = tf.abs(predicted - actual) / summ * 2.0
    return smape
def build_model(external_dim):
    #创建模型时   首先指定进行组合时的参数    将配置分别放进不同的区域中（就是相近性的长度 ，周期性的长度等）
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_period > 0 else None
    '''
    趋势性的数据暂时不要
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None
    '''
    #根据不同的配置定义残差神经网络模型  这个stresnet是定义好的，传入关于不同方面的配置，最后会返回根据参数组合好的模型
    model = stresnet(c_conf=c_conf, p_conf=p_conf,t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr) #接下来 定义学习率和损失函数值
    #sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse,smape])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def main():
    # ------------------------------------     配置模型和数据    --------------------------------------------#
    # load data
    print("loading data...")
    #开始加载数据   加载时指定各种参数，会根据传入的参数进行加载数据的分离。
    X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test= BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")
    model = build_model(external_dim)

    # ------------------------------------     第一次训练    --------------------------------------------#
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.bes'
                                        ''
                                        't.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')



    #————————————  配置网格搜索找参数 —————————#
    # model1=build_model(external_dim)
    # batch_size = [10, 20]
    # epochs = [10, 50, 100]
    # param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    # kflod = StratifiedKFold(n_splits=10)
    # grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1,scoring = smape,cv=kflod )
    # print('=' * 10)
    # print("training model...",Y_train.shape)
    # history = grid.fit(X_train, Y_train)
    # print("Best: %f using %s" % (history.best_score_, history.best_params_))
    # for params, mean_score, scores in history.grid_scores_:
    #     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

    # ————————————————————————————————————#

    print('=' * 10)
    print("training model...",Y_train.shape)
    history = model.fit(X_train, Y_train,
                         nb_epoch=nb_epoch,
                         batch_size=batch_size,
                         validation_split=0.1,
                         callbacks=[early_stopping, model_checkpoint],
                         verbose=1)

    '''
    这部分我想加入网格搜索，找到最佳的参数
    '''
    model.save_weights(os.path.join(
         'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))



    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')

    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 24, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f smape(normation):  %.6f ' %
          (score[0], score[1],score[2]))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f  smape(normation):  %.6f ' %
          (score[0], score[1],score[2]))

    showResult(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)))
#---------------------------------------      第二次训练        -----------------------------------------#

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')

    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[early_stopping,model_checkpoint], validation_data=(X_test, Y_test))
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 24, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f  smape(normation):  %.6f ' %
          (score[0], score[1],score[2]))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=1)
    print('Test score: %.6f rmse (norm): %.6f  smape(normation):  %.6f ' %
          (score[0], score[1],score[2]))


    showResult(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)))


if __name__ == '__main__':
    main()
