from __future__ import print_function
import pandas as pd
from datetime import datetime, timedelta
import time
import os

#对时间戳进行处理，将时间处理成  %Y-%m-%d-%H-%M类型的数据，相等于数据的重新规格化
def timestamp_str_new(cur_timestampes, T=48):
    os.environ['TZ'] = 'Asia/Shanghai'
    print('cur_timestampes:',cur_timestampes)  #如果已经包含'-'号说明已经处理过了
    if '-' in cur_timestampes[0]:
        return cur_timestampes
    ret = []
    for v in cur_timestampes:
        '''TODO
        Bug here
        对cur_timestampes中的每个时间点进行以下的处理，通过以下的方式进行规整化一下
        '''
        cur_sec = time.mktime(time.strptime("%04i-%02i-%02i" % (int(v[:4]), int(v[4:6]), int(v[6:8])), "%Y-%m-%d")) + (int(v[8:]) * 24. * 60 * 60 // T)
        curr = time.localtime(cur_sec)
        if v == "20151101288" or v == "2015110124":
            '''
            ctrl键看源代码发现，strftime的作用是将curr转换成前面定义的格式
            '''
            print(v, time.strftime("%Y-%m-%d-%H-%M", curr), time.localtime(cur_sec), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T)), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T) + 3600 * 25))
        ret.append(time.strftime("%Y-%m-%d-%H-%M", curr))
    return ret


def string2timestamp_future(strings, T=48):
    strings = timestamp_str_new(strings, T)
    timestamps = []
    for v in strings:
        #进行第二次时间戳转换，将其转成pandas中的timestamps格式
        year, month, day, hour, tm_min = [int(z) for z in v.split('-')]
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour, tm_min)))
    print('string2timestamp_future  timestamps:',timestamps)
    return timestamps


def string2timestamp(strings, T=24):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])
        #进行变换，按半小时计为时间戳，这是这个函数的作用
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))
    #z合理会有个转换，会将 半小时计为一戳，这是一种转换
    print('sss   strings :', strings, ' len: ', len(strings))
    print('sss   timestamps :',timestamps,' len: ',len(timestamps))
    return timestamps


def timestamp2string(timestamps, T=48):
    # timestamps = timestamp_str_new(timestamps)
    num_per_T = T // 24
    return ["%s%02i" % (ts.strftime('%Y%m%d'),
                        int(1+ts.to_datetime().hour*num_per_T+ts.to_datetime().minute/(60 // num_per_T))) for ts in timestamps]
    # int(1+ts.to_datetime().hour*2+ts.to_datetime().minute/30)) for ts in timestamps]

if __name__=='__main__':
    #作用是将  ['2014040101']->['2014-04-01-00-30']
    #timestamps=timestamp_str_new(['2014040101'],48)

    #作用是将  ['2014040101']->Timestamp('2014-04-01 00:30:00')  ,是一种变成时间戳格式的转换
    #timestamps=string2timestamp_future(['2014040101'],48)

    #作用是将  ['2014040101']->[Timestamp('2014-04-01 00:00:00')]  ,是一种变成整点时间戳格式的转换，24小时那种
    #timestamps=string2timestamp(['2014040101'],48)

    #作用是将  Timestamp('2014-04-01 00:00:00')->['2014040101']  ,是一种方式的回转，将timestamp转成字符串式文件
    timestamps=timestamp2string([pd.Timestamp('2014-04-01 00:00:00')],48)
    print(timestamps)
