import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
'''
这里有些代码 是对数据空值的 一种填充方式，  还有  是对一种排序，因为这是我的一个要求，让数据按时间和站点进行组合，将每个时间段的35个站点数据都组合在一起。。。
'''
# data = pd.read_csv('E:/final_merge_aq_grid_meo_with_weather.csv')
# # print(data.describe())
# data = data.fillna(method='ffill')   #这里的fillna是对空值的一种填充方式， ffill是使用前一个非缺失值来填充缺失值。
# # data.dropna()
# data.to_csv('D:/final_merge_aq_grid_meo_with_weather.csv',index=False)
#
# # data = pd.read_csv('D:/assets/output/final_merge_aq_grid_meo.csv')
# # pd.to_datetime(data['utc_time'])
# # # data = data.sort_values(by = ['utc_time'])
# # data = data.loc[data['utc_time'] >= '2017-01-31 00:00:00']
# # data.to_csv('D:/assets/temp.csv',index=False)
# # data = data.loc[data['utc_time'] <= '2018-01-31 23:00:00']

# data = pd.read_csv('D:/final_merge_aq_grid_meo_with_weather.csv')
# data = data.sort_values(by = ['utc_time', 'stationId_aq'])
# data.to_csv('D:/final_merge_aq_grid_meo_with_weather.csv',index=False)
#
# data2 = data[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
# data3 = data[['stationId_aq', 'utc_time']]
# frames = [data3, data2]
#
# result = pd.concat(frames, axis=1)
# result.to_csv('D:/final_merge_aq_grid_meo_with_weather_standardization.csv',index=False)
#
# print(data.describe())
data = pd.read_csv('D:/final_merge_aq_grid_meo_with_weather_standardization.csv')
print(data)