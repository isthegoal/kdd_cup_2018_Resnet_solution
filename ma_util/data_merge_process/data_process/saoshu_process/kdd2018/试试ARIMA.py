from __future__ import absolute_import, division, print_function

import sys
import os

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')



#now ,let's start!
data = pd.read_csv('F:/BaiduNetdiskDownload/kDD 2018/Beijing/beijing_17_18_aq.csv')
data.dropna()
aotizhongxin_aq = data.loc[data['stationId'] == 'aotizhongxin_aq']
aotizhongxin_aq.index = pd.to_datetime(aotizhongxin_aq['utc_time'])
day = aotizhongxin_aq['PM10'].resample('6H').mean()
day.dropna()
train = day['2017-01-01':'2017-11-01']


stock_diff = train.diff()
stock_diff = stock_diff.dropna()

# plt.figure()
# plt.plot(stock_diff)
# plt.title('一阶差分')
# plt.show()
#
# #q=4
# acf = plot_acf(stock_diff, lags=20)
# plt.title("ACF")
# plt.show()
#
# #p = 3 or 4
# pacf = plot_pacf(stock_diff, lags=20)
# plt.title("PACF")
# plt.show()

model = ARIMA(train, order=(1, 1, 1), freq='6H')
result = model.fit()
pred = result.predict('2017-10-01', '2017-12-01',dynamic=True, typ='levels')
print (pred)

# aotizhongxin_aq['diff_1'] = aotizhongxin_aq['PM10'].diff(1)
# aotizhongxin_aq['diff_2'] = aotizhongxin_aq['diff_1'].diff(1)
# print(data.dtypes)

# n_sample = aotizhongxin_aq.shape[0]
#
# n_train = int(0.95*n_sample) + 1
# n_forecast = n_sample - n_train
# ts_train = aotizhongxin_aq.iloc[:n_train]['PM10']
# ts_test = aotizhongxin_aq.iloc[n_train:]['PM10']
# print(ts_train.dtypes)
# print(ts_test.shape)
# train.plot(figsize=(12,8))
# plt.legend(bbox_to_anchor=(1.25, 0.5))
# plt.title("Stock Close")
# plt.show()

# pm10_diff = aotizhongxin_aq['PM10'].diff(1)
# plt.figure()
# plt.plot(pm10_diff)
# plt.title('一阶差分')
# plt.show()
#
# acf = plot_acf(pm10_diff, lags=10)
# plt.title('ACF')
# plt.show()
#
# pacf = plot_pacf(pm10_diff, lags=20)
# plt.title("PACF")
# plt.show()
#
# model = ARIMA(ts_train, order=(0, 1, 0))
# result = model.fit()
#
# pred = result.predict('20170801', '20180131')
# print(pred)
# def tsplot(y, lags=None, title='', figsize=(24, 18)):
#     fig = plt.figure(figsize=figsize)
#     layout = (2, 2)
#     ts_ax = plt.subplot2grid(layout, (0, 0))
#     hist_ax = plt.subplot2grid(layout, (0, 1))
#     acf_ax = plt.subplot2grid(layout, (1, 0))
#     pacf_ax = plt.subplot2grid(layout, (1, 1))
#
#     y.plot(ax=ts_ax)
#     ts_ax.set_title(title)
#     y.plot(ax=hist_ax, kind='hist', bins=25)
#     hist_ax.set_title('Histogram')
#     smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
#     smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
#     [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
#     # sns.despine()
#     plt.show()
#     fig.tight_layout()
#     return ts_ax, acf_ax, pacf_ax
#
# tsplot(ts_train, title='A Given Training Series', lags=20)