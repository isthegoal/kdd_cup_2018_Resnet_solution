import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/beijing_17_18_aq.csv')
# print(data.head())
print(data.tail())


# print(len(data['stationId'].unique()))
#
# aotizhongxin_aq = data.loc[data['stationId'] == 'aotizhongxin_aq']
# aotizhongxin_aq.index = pd.to_datetime(aotizhongxin_aq['utc_time'])
#
# print(aotizhongxin_aq.head())
#
# aotizhongxin_aq['PM2.5'].plot(style='r')
# # aotizhongxin_aq['PM2.5'].rolling(window=24).mean().plot(style='b')
# plt.show()
# fig = plt.figure(1, figsize=[10, 10])
# plt.ylabel('浓度：(微克/立方米)')
# plt.xlabel('时间：月份')
# plt.title('aotizhongxin_aq')
# key = ['PM2.5', 'PM10', 'O3']
# for i in key:
#     plt.plot(aotizhongxin_aq['utc_time'] , aotizhongxin_aq[i])
# plt.legend(key)
#
# plt.show()
