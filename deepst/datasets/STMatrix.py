from __future__ import print_function
import os
import pandas as pd
import numpy as np

from deepst.datasets import load_stdata
from deepst.config import Config
from deepst.utils import string2timestamp

#对时空矩阵的处理
class STMatrix(object):
    """docstring for STMatrix"""
    #时空矩阵初始化   创建构造时空矩阵  重要的索引信息就是时间信息
    def __init__(self, data, timestamps,data_air,pred_48_air, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)  #时间长度和数据信息相等
        self.data = data
        self.timestamps = timestamps
        self.data_air=data_air
        self.pred_48_air=pred_48_air
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()
    #为时空矩阵创建索引
    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            #print('pd_timestamps下的',',i:',i,',ts:',ts)
            self.get_index[ts] = i   #构建字典信息， 将时间戳作为索引
    #检查矩阵内数据的完整性   self是用来构造时空矩阵的数据
    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            #这里的意思是一种对时间戳缺失的检验，如果一时刻下一时间戳不是半小时后的数据，说明丢失了时间戳，这是对数据的检验
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0
    #获取指定时间段的对应矩阵数据
    def get_matrix(self, timestamp):  #可以通过得到 时间戳所对应的位置索引找到索引所对应的这个时间的流量数据
        return self.data[self.get_index[timestamp]]

    #获取指定时间对应的未来的50个小时的天气信息
    def get_air_48pred_matrix(self,timestamp):
        #print('the timestamp:',timestamp)
        return self.pred_48_air[self.get_index[timestamp]]


    #这里获得是  要预测的结果，我们每次使用的一个预测时刻点下的6个时间点的数据
    def get_result_matrix(self, timestamp):
        return self.data_air[self.get_index[timestamp]:self.get_index[timestamp]+50]
    #---------     针对  目标35*3列表创造辅助函数  -----------------#
    def get_air_matrix(self, timestamp):  # 可以通过得到 时间戳所对应的位置索引找到索引所对应的这个时间的流量数据
        return self.data_air[self.get_index[timestamp]]

    # 这里获得是  要预测的结果，我们每次使用的一个预测时刻点下的6个时间点的数据
    def get_air_result_matrix(self, timestamp):
        return self.data_air[self.get_index[timestamp]:self.get_index[timestamp] + 50]
    #---------------------------------------------------------#
    def save(self, fname):
        pass

    def check_it(self, depends):
        #检查有效性，  对所有depends进行检测， 如果depends是不存在的时间戳，说明这个数据是无效的，多出的
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True
    def create_dataset(self, len_closeness=6, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        这个函数很猛，这里会对所有的时间数据和 流量数据进行组装，将他们分别包装到趋势性、周期性、邻近性的三个数组中，作为模型
        的分别输入使用（这里指定参数是以三个为一个相近性和趋势性 ，7个为一个周期，TrendInterval是趋势计算的长度）
        """
        # offset_week = pd.DateOffset(days=7)
        '''
         Each offset specify a set of dates
    that conform to the DateOffset.  For example, Bday defines this
    set to be the set of dates that are weekdays (M-F).  To test if a
    date is in the set of a DateOffset dateOffset we can use the
    onOffset method: dateOffset.onOffset(date).
        '''
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT=[]
        X48_air=[]
        Y = []
        timestamps_Y = []
        # 设置针对不同时间特性的依赖性，  第一个是周围三天，第二个是更大间隔三天len_period性，第三个是最大的趋势性间隔21天的TrendInterval
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)],
                   range(0, 50)
                   ]
        #TrendInterval * len_trend是21天，PeriodInterval * len_period是三天，  取最大是因为要从这里开始训练，之前部分的训练数据是不足的。
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        #这里预测 后6个小时 的 所以这里判断条件得改一下
        while i < len(self.pd_timestamps)-50:
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                #对所有时间戳进行检查
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])
                # if(str(self.pd_timestamps[i])[-8:-1]!='22:00:0'):
                #
                #     Flag=False
            if Flag is False:
                i += 1
                continue
            #print('我需要的筛选的时间格式为：', str(self.pd_timestamps[i]))
            #将不同的depend参与计算，分别得到三个不同时间特性的矩阵
            x_c = [self.get_matrix(self.pd_timestamps[i]- j*offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i]- j*offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i]- j*offset_frame) for j in depends[2]]
            x_48_air=[self.get_air_48pred_matrix(self.pd_timestamps[i+j]) for j in depends[3]]
            y = self.get_result_matrix(self.pd_timestamps[i])
            #这里相比与 以前去除了 vstack的作用，  这样之后维度上达到了 自己想要的效果。。。
            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            X48_air.append(x_48_air)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
            #print('pd_timestamps 的长度：',len(self.pd_timestamps))
            #print('i 的变化：',i)
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        X48_air=np.asarray(X48_air)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape,"Y shape:", Y.shape,' the air pred',X48_air.shape)
        #返回三个矩阵和所有的时间戳      y是单独点的时间戳
        return XC, XP,XT, Y,timestamps_Y

if __name__ == '__main__':
    pass
