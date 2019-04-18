import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''
这里是对数据进行了标准化处理，  我看他的方式挺好，先针对数字列上的特征值进行标准化处理（转成-1到1之间）， 之后再于非数值拼接并保存。。。
'''

# data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/final_merge_aq_grid_meo.csv')
# data = data.sort_values(by = ['utc_time', 'stationId_aq'])
# data.to_csv('D:/final_merge_aq_grid_meo.csv', index=False)
# print(data[0: 35])
# print(data[35:69])

# data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/final_merge_aq_grid_meo_with_weather.csv')
# data = data.sort_values(by = ['utc_time', 'stationId_aq'])
# data.to_csv('D:/final_merge_aq_grid_meo_with_weather.csv', index=False)

data = pd.read_csv('D:/final_merge_aq_grid_meo_with_weather.csv')
# data = data[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]
#
# peason = data.corr()
# print(peason)
# plt.pcolor(peason)
# plt.show()
data2 = data[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
data3 = data[['stationId_aq', 'utc_time']]
frames = [data3, data2]  #在列上进行拼接。。。    还有很多链接方式的选择，左右、外链接等方式，全在书上，方式很多。。。

result = pd.concat(frames, axis=1)
print(result.head())