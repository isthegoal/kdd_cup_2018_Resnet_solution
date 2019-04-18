'''
根据模板合并数据
beijing_17_18_aq.csv
beijing_17_18_meo.csv
模板
merage_aq_meo_beijing_template.csv
输出文件
assets/output/merage_aq_meo/merage_aq_meo_beijing.csv
'''

import pandas as pd
from math import*
import matplotlib.pyplot as plt
from ma_util.data_merge_process.data_process.tools import *



'''
这段程序是按  经纬度对  质量检测站和气象站的信息进行融合，     按照他们的意思这是最后一步的 将气象站的数据才利用起来， 但那是我看这里
他用的beijing_17_18_aq初始的信息,  不该是已经组合好的数据再开始进行  融入气象数据的吗？？？ 
'''


            #print(df_aq_grid_drop.loc[index].values[0])
# 根据经纬度聚类质量观测站 和 气象观测站

# 读入数据文件
aq_data = pd.read_csv("assets/data/beijing_17_18_aq.csv")
meo_data = pd.read_csv("assets/data/beijing_17_18_meo.csv")
# 质量观测站经纬度对应表数据
aq_station_data = pd.read_csv("assets/data/Beijing_AirQuality_Stations_cn_data.csv")

# 将气象经纬度观测点对应数据 抽离成一个字典形式用于连接两张表
station_id_meo_group = meo_data.groupby(meo_data['station_id'])
station_id_meo_dic = {}
for name,group in station_id_meo_group:
    lng_lat_list = []
   # print(name)
    longitude = group[0:1].longitude.values[0]
    latitude =  group[0:1].latitude.values[0]
    lng_lat_list.append(longitude)
    lng_lat_list.append(latitude)
    station_id_meo_dic[name] = lng_lat_list
   # print(station_id_meo_dic[name])

# 开始匹配经纬度距离最近的气象观测站和质量观测站
print("===================输出匹配后的观测站列表，第二个是质量站，第一个是气象站")
station_id_aq_group = aq_station_data.groupby(aq_station_data['stationId'])
aq_meo_match_dict = {}
for name,group in station_id_aq_group:
    aq_longitude = group[0:1].longitude.values[0]
    aq_latitude = group[0:1].latitude.values[0]
    #print(name,aq_longitude,aq_latitude)
    cur_min_meo_name = None
    cur_min_dis = None
    for k,v in station_id_meo_dic.items():
        meo_station_name = k
        meo_longitude = v[0]
        meo_latitude = v[1]
        # print("----------------")
        # print(meo_longitude,meo_latitude)
        dis = Distance1(aq_latitude,aq_longitude,meo_latitude,meo_longitude)
        if cur_min_meo_name is None:
            cur_min_meo_name = meo_station_name
            cur_min_dis = dis
        else:# 冒泡排序把最小的放到进栈位
            if cur_min_dis > dis:
                cur_min_meo_name = meo_station_name
                cur_min_dis = dis

    aq_meo_match_dict[name] = cur_min_meo_name

for k,v in aq_meo_match_dict.items():
    # if k == "aotizhongxin_aq":
    #     pass
    print(k,v)

# 开始合并流程
print("==================开始合并流程===============")
print("==================处理气象数据、增加对应的质量站列=========开始======")
df_meo = pd.DataFrame(meo_data,index=None,columns=['station_id','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather'])
df_meo.columns = ['stationId_meo','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather']


print("==================处理气象数据、增加对应的质量站列=========结束======")
print(df_meo.tail(100))

print("==================处理质量数据集、删减到需要合并的数据集===开始============")
df_aq = pd.DataFrame(aq_data,index=None,columns=['stationId','utc_time','NO2','CO','SO2','PM2.5','PM10','O3'])
df_aq.columns = ['stationId_aq','utc_time','NO2','CO','SO2','PM2.5','PM10','O3']
df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])
aq_series_copy = df_aq['stationId_aq'].copy(True)
df_aq.insert(1,'stationId_meo',aq_series_copy)
for k,v in aq_meo_match_dict.items():
    #df_meo = df_meo.copy()
    aq_name = k
    meo_name = v
    # if aq_name == "aotizhongxin_aq":
    #     print(aq_name)
    # print(k,v)
    df_aq.stationId_meo[df_aq.stationId_aq == aq_name] = meo_name
#print(df_aq)


print("==================开始处理质量数据集、删减到需要合并的数据集===结束===============")
print("==================合并的数据集========开始=======")
merge_data = pd.merge(df_meo,df_aq,on=['stationId_meo','utc_time'],how='right')
df_merge = pd.DataFrame(merge_data)

print("==================合并的数据集========结束=======")
print('天气数据总记录数===》',df_aq.shape[0])
print('合并后总记录数===》',df_merge.shape[0])


print("==================处理插值========开始=======")
# 全字段插值方法
def interpolation_all_columns():
    pass

print("==================处理插值========结束=======")

print("==================合并的数据集========结束=======")
print("==================写入文件merge_aq_meo.csv========开始=======")
df_merge.to_csv('assets/output/merge_aq_meo/merge_aq_meo.csv',index=False)
print("==================写入文件merge_aq_meo.csv========结束=======")

# for meo_station_name,data_point in station_id_meo_dic.items():
#     meo_longitude = data_point[0]
#     meo_latitude = data_point[1]
#     #print(name,aq_longitude,aq_latitude)
#     cur_min_aq_name = None
#     cur_min_dis = None
#     for name, group in station_id_aq_group:
#         aq_station_name = name
#         aq_longitude = group[0:1].longitude.values[0]
#         aq_latitude = group[0:1].latitude.values[0]
#         # print("----------------")
#         # print(meo_longitude,meo_latitude)
#         dis = Distance1(aq_latitude,aq_longitude,meo_latitude,meo_longitude)
#         if cur_min_aq_name is None:
#             cur_min_aq_name = aq_station_name
#             cur_min_dis = dis
#         else:# 冒泡排序把最小的放到进栈位
#             if cur_min_dis > dis:
#                 cur_min_aq_name = aq_station_name
#                 cur_min_dis = dis
#
#     aq_meo_match_dict[meo_station_name] = cur_min_aq_name