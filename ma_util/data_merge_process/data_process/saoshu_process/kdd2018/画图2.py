import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing_historical_meo_grid.csv')
print(len(data))
data2 = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/beijing_17_18_meo.csv')
data3 = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/2.csv')
print(data3)
aotizhongxin_aq = data.loc[data['stationId_aq'] == 'aotizhongxin_aq']
print(aotizhongxin_aq.describe())
print(data.head())
print(data.describe())
print(data['stationId_meo'].unique())

jw = data.loc[:650, ['longitude', 'latitude']]
jw2= data2.loc[:, ['longitude', 'latitude']]
jw2 = jw2.drop_duplicates()
# print(jw2)
#
plt.figure()
plt.scatter(jw['longitude'], jw['latitude'], c='r')
# plt.scatter(jw2['longitude'], jw2['latitude'], c='b')
plt.scatter(data3['longitude'], data3['latitude'], c='y')
key = ['Grid',  'Observation point']
plt.legend(key)
plt.show()