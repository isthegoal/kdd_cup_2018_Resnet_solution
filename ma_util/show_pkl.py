# encoding: utf-8
import sys
import pickle
import numpy as np
#遇到  no moduled named tkinter 解决方法是: apt-get install python-tk
import matplotlib.pyplot as plt
import tensorflow as tf
# def view(fname):
#     pkl = pickle.load(open(fname, 'rb'))
#     for ke in pkl.keys():
#          print('=' * 10)
#          print(ke)
#          print(pkl[ke])
# view('/home/fly/PycharmProjects/DeepST-master/scripts/papers/AAAI17/BikeNYC/RET/c3.p4.t4.resunit4.lr0.0002.cont.history.pkl')

def showResult(fname):
    print('展示出曲线')
    pkl = pickle.load(open(fname, 'rb'))
    for ke in pkl.keys():
        if ke=='loss':
            loss_list=pkl[ke]
        if ke=='val_loss':
            val_loss_list=pkl[ke]
        if ke=='rmse':
            rmse_list=pkl[ke]
        if ke=='val_rmse':
            val_rmse_list=pkl[ke]
    x = np.arange(len(loss_list))
    plt.plot(x, loss_list, linestyle="--", color="orange",label='train loss value')
    plt.plot(x, val_loss_list, linestyle="--", color="red",label='test loss value')
    plt.title("the loss value")
    plt.xlabel("number")
    plt.ylabel("loss value")
    plt.legend(["train loss value", "test loss value"], loc="upper right")
    plt.show()

    plt.plot(x, rmse_list, linestyle="--", color="orange",label='train rmse value')
    plt.plot(x, val_rmse_list, linestyle="--", color="red",label='test rmse value')
    plt.title("the rmse value")
    plt.xlabel("number")
    plt.ylabel("rmse value")
    plt.legend(["train rmse value", "test rmse value"], loc="upper right")
    plt.show()

#通过替换路径，展示不同的图线效果     rmse是均方差误差   loss(mse是方根误差)和普通的rmse不同    第一个训练设置了早停，所以会在一半就中断了
'''
MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。

均方误差:均方根误差是均方误差的算术平方根
'''
#showResult('/home/fly/PycharmProjects/DeepST-KDD/scripts/AirPrediction/RET/c6.p4.t4.resunit12.lr0.0002.cont.history.pkl')