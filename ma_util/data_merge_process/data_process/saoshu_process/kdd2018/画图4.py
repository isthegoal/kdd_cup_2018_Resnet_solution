import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans


# place = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/place.csv')
df_aq = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/beijing_17_18_aq.csv')
df_aq.columns = ['stationId_aq','utc_time','NO2','CO','SO2','PM2.5','PM10','O3']
df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])
print(df_aq)

# #空气质量数据有了经纬度 stationId  utc_time  PM2.5   PM10    NO2   CO   O3  SO2 longitude  latitude
# result = pd.merge(data, place, on=['stationId'])
# print(result.head())
#
# grid = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing_historical_meo_grid.csv')

# df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])