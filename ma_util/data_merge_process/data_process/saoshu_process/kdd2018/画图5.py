import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


'''
这部分应该包含了将天气的字符型数据 准成数据的 转换代码
缺失直插值的方法（线性插值，也就一句话的事）  以及  去除空值的代码
'''

# data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing_historical_meo_grid.csv')
# jw = data.loc[:6499, :]
# jw = jw.sort_values(by = ['stationName', 'utc_time'])
# print(jw)

# data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/final_merge_aq_grid_meo.csv')
# data = data.interpolate()
# data.to_csv('D:/final_merge_aq_grid_meo.csv', index=False)
data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/final_merge_aq_grid_meo_with_weather.csv')
# data = data.loc[data['utc_time'] >= '2017-1-30 16:00:00']
# data = data.dropna()

#   这意思是去除所有   weather为空的数据行，  这里 NAN也就是空值，    ～是一种去除的方法，学习到了，这里去除的原因是因为有连续20天没有数据，哈哈。。。
data = data[~data['weather'].isnull()]
#loc是一种  横纵向指定的定位方式    定位之后会将  天气补成 1～10之间的数字
data.loc[data["weather"] == "Sunny/clear", "weather"] = 1
data.loc[data["weather"] == "Haze", "weather"] = 2
data.loc[data["weather"] == "Snow", "weather"] = 3
data.loc[data["weather"] == "Fog", "weather"] = 4
data.loc[data["weather"] == "Rain", "weather"] = 5
data.loc[data["weather"] == "Dust", "weather"] = 6
data.loc[data["weather"] == "Sand", "weather"] = 7
data.loc[data["weather"] == "Sleet", "weather"] = 8
data.loc[data["weather"] == "Rain/Snow with Hail", "weather"] = 9
data.loc[data["weather"] == "Rain with Hail", "weather"] = 10
data = data.interpolate() #这是补全缺失值的方法，会使用线性 插值法 补全缺失值。。。
data = data[['stationId_aq', 'utc_time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]
data.to_csv('D:/final_merge_aq_grid_meo_with_weather.csv', index=False)
print(data.describe())






# print(data.head())
# print(data.describe())
# data = data.sort_values(by=[])
