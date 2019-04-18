import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''
这里面包含了对异常天气数值的  处理方式中   dropna方式是对整个为空的都去除，  所以从这里又损失数据了，  这里 我听他说的  对于异常数据直是应该进行线性插值的，所以我看他这里
代码有点混乱，来源于他的代码习惯不好， 正规的思想是要对异常数据 都设置为空，对于空数据再使用线性插值的方法进行处理。。   他的代码不太好，自己可以重整下。。。
'''


data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/beijing_17_18_meo.csv')
print(data)
# print(data.describe())
# print(data.loc[data['temperature'] > 100])
# print(data.loc[data['pressure'] > 2000])
# print(data.loc[data['humidity'] > 100, 'humidity'])
# print(len(data.loc[data['wind_direction'] > 360, 'wind_direction']))
# print(data.loc[data['wind_speed'] > 50, 'wind_speed'])



# #处理温度的异常值，温度大于100为观测错误
# temp = data['temperature'].dropna()
# temp = np.array(temp)
# temp[temp > 100] = 0
# mean = np.mean(temp)
# data.loc[data['temperature'] > 100, 'temperature'] = mean
# data['temperature'].fillna(mean)
# # print(data['temperature'])
#
# #处理压力的异常值，温度大于2000为观测错误
# temp = data['pressure'].dropna()
# temp = np.array(temp)
# temp[temp > 2000] = 0
# mean = np.mean(temp)
# data.loc[data['pressure'] > 2000, 'pressure'] = mean
# data['pressure'].fillna(mean)
# # print(data['temperature'])
#
# #处理湿度的异常值，温度大于100为观测错误
# temp = data['humidity'].dropna()
# temp = np.array(temp)
# temp[temp > 100] = 0
# mean = np.mean(temp)
# data.loc[data['humidity'] > 100, 'humidity'] = mean
# data['humidity'].fillna(mean)
# # print(data['humidity'])
#
# #处理风向的异常值,风向大于360为观测错误，但是风向越有5000异常值
# temp = data['wind_direction'].dropna()
# temp = np.array(temp)
# temp[temp > 360] = 0
# mean = np.mean(temp)
# data.loc[data['wind_direction'] > 360, 'wind_direction'] = mean
# data['wind_direction'].fillna(mean)
# print(data['wind_direction'])
#
#
# #处理风速的异常值，温度大于50为观测错误
# temp = data['wind_speed'].dropna()
# temp = np.array(temp)
# temp[temp > 50] = 0
# mean = np.mean(temp)
# data.loc[data['wind_speed'] > 50, 'wind_speed'] = mean
# data['wind_speed'].fillna(mean)
# # print(data['humidity'])
#
# shunyi_meo = data.loc[data['station_id'] == 'shunyi_meo']
# shunyi_meo['utc_time'] = pd.to_datetime(shunyi_meo['utc_time'])
#
# fig = plt.figure(1, figsize=[10, 10])
# plt.ylabel('temperature')
# plt.xlabel("time")
#
# plt.title('aotizhongxin_aq')
# key = ['pressure']  #longitude：经度， latitude：纬度， utc_time：时间（以小时为单位）， temperature：温度，
#                                 #pressure：压力， humidity：湿度， wind_direction：风向， wind_speed：风速， weather：天气
# for i in key:
#     plt.plot(shunyi_meo['utc_time'], shunyi_meo[i])
# plt.legend(key)
# plt.show()
