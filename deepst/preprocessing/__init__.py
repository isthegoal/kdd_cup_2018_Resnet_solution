import pandas as pd
import numpy as np
from copy import copy
import time
import json
import urllib
from datetime import datetime
from datetime import timedelta #
from chinese_calendar import is_workday, is_holiday
# from temporal_contrast_normalization import TemporalConstrastNormalization
# from personal_temporal_contrast_normalization import PersonalTemporalConstrastNormalization
from deepst.preprocessing.minmax_normalization import MinMaxNormalization
from deepst.utils import string2timestamp
def is_beijing_holiday(need_data):
    #print('i need panduan ',need_data)
    beijing_holiday_list=['2017-01-01', '2017-01-02','2017-01-27', '2017-01-28', '2017-01-29','2017-01-30', '2017-01-31', '2017-02-01','2017-02-02', '2017-04-02', '2017-04-03','2017-04-04', '2017-05-01', '2017-05-28','2017-05-29', '2017-05-30', '2017-10-01','2017-10-02', '2017-10-03', '2017-10-04','2017-10-05', '2017-10-06', '2017-10-07','2017-10-08', '2018-01-01', '2018-02-15','2018-02-16', '2018-02-17', '2018-02-18','2018-02-19','2018-02-20', '2018-02-21','2018-04-05', '2018-04-06', '2018-04-07','2018-04-30', '2018-05-01']
    if str(need_data)[:10] in beijing_holiday_list:
        return  True
    else:
        return  False
#对之前方法的改进，包括对节假日的识别，如果是工作日标识为1，如果是假日在第9个位置上标识为1.
#下面是针对节假日判断和 工作日判断的一种可行的方法(很棒的可以完全进行判断了，最后使用array保存，为了好处理)。   这里的9函数，拟定的输入就是字符串列表把。
def timestamp9vec(timestamps):
    '''
    这里的一些思考，weekday  能够返回周几，   is_holiday能判断国内的节假日
    我现在加入一些时间上的延迟后一天，后两天，所以记住预测时候  要是前一天的日期。。。
    '''
    print('timestamps  timestamp9vec:',timestamps)
    #datetime.timedelta(926,56700)
    # tm_wday range [0, 6], Monday is 0             weekday
    #原论文是这样处理方式  vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [(datetime.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d')+timedelta(1)).weekday() for t in timestamps]  # python3

    #date_time = datetime.datetime.strptime(str, '%Y-%m-%d')
    #print('timestamps[1])[:8 :',str(timestamps[1][:8]))
    bool_holiday=[is_beijing_holiday(datetime.strptime(str(i[:8], encoding='utf-8'), '%Y%m%d')+timedelta(1)) for i in timestamps]  #先转成date类型,再进行是否为节假日的判断



    #------------------------------------------  加入第二天   ----------------------------#
    vec2 = [(datetime.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d') + timedelta(2)).weekday() for t in timestamps]  # python3

    # date_time = datetime.datetime.strptime(str, '%Y-%m-%d')
    # print('timestamps[1])[:8 :',str(timestamps[1][:8]))
    bool_holiday2 = [is_beijing_holiday(datetime.strptime(str(i[:8], encoding='utf-8'), '%Y%m%d') + timedelta(2)) for i in timestamps]  # 先转成date类型,再进行是否为节假日的判断

    #------------------------------------------------------------------------------------#
    #print(bool_holiday)
    #vec = [time.strptime(t[:8].decode("ASCII"), '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    bool_index=0
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5: 
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        if bool_holiday[bool_index] == True:
            v.append(1)
        else:
            v.append(0)

        #-------------    加入第二天的信息   附加到原来的列表中去 ------------------#
        v.extend([0,0,0,0,0,0,0])
        v[9+vec2[bool_index]]=1
        if vec2[bool_index]>= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        if bool_holiday2[bool_index] == True:
            v.append(1)
        else:
            v.append(0)
        #print('timestamps :',timestamps[bool_index],'    v:',v)
        # ---------------------------------------------------#
        ret.append(v)
        bool_index+=1
    return np.asarray(ret)


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0     下面的函数能够获取到具体所在的星期几
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    #vec = [time.strptime(t[:8].decode("ASCII"), '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    #根据星期几 和 是否星期天、节假日进行组合。
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)



#去除不包含48时间戳的日，放在days_incomplete中的数据就不要了，放弃不完整数据天的数据
#这里的函数有问题，一天周期是48（[8:]) == T），不是24，还没进行转化，所以这一不需要进行取出，因此这个程序是有问题的，所有的不完整天都取出不了
#思路还可以...
def remove_incomplete_days(data, timestamps, T=24):
    print('timestamps:：：：',timestamps)
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        #print(' ',timestamps[1][8:],' ',timestamps[1][:8])
        if int(timestamps[i][8:]) != 0:
            i += 1
        elif i+T < len(timestamps) and int(timestamps[i+T][8:]) == 0:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    #这是自己附加的一条语句，因为正规的方法中最后一天并没有将输入融加进去，而我们是需要的
    days.append(timestamps[-1][:8])
    print("append days: ", days[-1])
    print("incomplete dayss: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def split_by_time(data, timestamps, split_timestamp):
    # divide data into two subsets:
    # e.g., Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    assert(len(data) == len(timestamps))
    assert(split_timestamp in set(timestamps))

    data_1 = []
    timestamps_1 = []
    data_2 = []
    timestamps_2 = []
    switch = False
    for t, d in zip(timestamps, data):
        if split_timestamp == t:
            switch = True
        if switch is False:
            data_1.append(d)
            timestamps_1.append(t)
        else:
            data_2.append(d)
            timestamps_2.append(t)
    return (np.asarray(data_1), timestamps_1), (np.asarray(data_2), timestamps_2)


def timeseries2seqs(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y

def timeseries2seqs_meta(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    avail_timestamps = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            avail_timestamps.append(raw_ts[idx[i+length]])
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y, avail_timestamps


def timeseries2seqs_peroid_trend(data, timestamps, length=3, T=48, peroid=pd.DateOffset(days=7), peroid_len=2):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            # period
            target_timestamp = timestamps[i+length]

            legal_idx = []
            for pi in range(1, 1+peroid_len):
                if target_timestamp - peroid * pi not in timestamp_idx:
                    break
                legal_idx.append(timestamp_idx[target_timestamp - peroid * pi])
            # print("len: ", len(legal_idx), peroid_len)
            if len(legal_idx) != peroid_len:
                continue

            legal_idx += idx[i:i+length]

            # trend
            x = np.vstack(data[legal_idx])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_3D(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = data[idx[i:i+length]].reshape(-1, length, 32, 32)
            y = np.asarray([data[idx[i+length]]]).reshape(-1, 1, 32, 32)
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def bug_timeseries2seqs(data, timestamps, length=3, T=48):
    # have a bug
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            breakpoints.append(i)
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - 3):
            x = np.vstack(data[idx[i:i+3]])
            y = data[idx[i+3]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y

if __name__=='__main__':
    timelist=['2018040702','2018040802','2018041802']
    arr=timestamp9vec(timelist)
    print('arr:',arr)