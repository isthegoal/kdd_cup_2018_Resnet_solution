'''
使用了merge_grid_meo的合并卫星的气象数据
合并beijing_17_18_aq质量站数据
++++++++++++++++++++++++++++++++++++++++++++
'''

import pandas as pd
from math import*
import matplotlib.pyplot as plt
from ma_util.data_merge_process.data_process.tools import *



'''
这里是对两个文件进行合并，beijing_17_18_aq   和  merge_grid_meo   ，这里 merge_grid_meo应该是已经按经纬度抽取过的天气信息，这里算是比较靠后的融合方式。 
从功能上来看，这应该是 在 'merge_meo_gird'之后进行的程序， 起作用是因为上个文件根据位置   聚集天气时并没有一次性把35个站点的空气监测信息也放进去，所以此处才需要 再一步
进行融合，其实我感觉这一步都是多余的，直接 上一步处理完就行了，就是融合而已。。。


终于理解套路了，理解的难点就在于这个    
df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])
上。

这几个文件是
（1）先用 merge_aq_meo(别被取名误导)做下数据补全，也就是   使用所有的 气象站点数据去得到 35个站点的气象数据， 这是一个获得所有站点的气象数据的方式，会将所有数据合在一起，但是这里获得的
组合方式肯定是会有时间缺失的。
（2）第二次是 结合文件  merge_aq_meo和merge_meo_gird 两个文件处理过之后，将两者结合能够得到更加比较完善的35个站点的气象数据，但是这和第一种方式(1)肯定会有数据是时间点重合的，因为
本来  气象站数据和网格数据的时间就是重合的， 所以这里才需要使用drop_duplicates进行一次去重处理，保留第一次不重复的，但是这种保留其实并没有对谁更有用进行筛选。。

思考：  有个想法是  网格上的天气数据应该是全的把，不全的好像应该是 空气质量信息， 其实全不全也不一定，因为他们有些操作是直接删除一次数据的，导致时间点数据没有了，爱，有点麻烦，
不过还好， 新的数据没有多少缺少的，多按照自己的思路整下吧

'''


            #print(df_aq_grid_drop.loc[index].values[0])
# 根据经纬度聚类质量观测站 和 气象观测站

# 读入数据文件
aq_data = pd.read_csv("assets/data/beijing_17_18_aq.csv")
meo_data = pd.read_csv("assets/output/merge_aq_meo/merge_grid_meo.csv")

# 开始合并流程
print("==================开始合并流程===============")
print("==================处理气象数据、增加对应的质量站列=========开始======")
df_meo = pd.DataFrame(meo_data,index=None,columns=['stationName','utc_time','temperature','pressure','humidity','wind_direction','wind_speed/kph'])
df_meo.columns = ['stationId_aq','utc_time','temperature','pressure','humidity','wind_direction','wind_speed/kph']


print("==================处理气象数据、增加对应的质量站列=========结束======")
#print(df_meo.tail(100))

print("==================处理质量数据集、删减到需要合并的数据集===开始============")
df_aq = pd.DataFrame(aq_data,index=None,columns=['stationId','utc_time','NO2','CO','SO2','PM2.5','PM10','O3'])
df_aq.columns = ['stationId_aq','utc_time','NO2','CO','SO2','PM2.5','PM10','O3']
df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])#删除两个列组合列在一起的重复项   估计没用上

print("==================开始处理质量数据集、删减到需要合并的数据集===结束===============")
print("==================合并的数据集========开始=======")
merge_data = pd.merge(df_meo,df_aq,on=['stationId_aq','utc_time'],how='right')   #根据站点id和时间进行右外连接，   意思是df_aq作为连接的主体键列。
df_merge = pd.DataFrame(merge_data)

print("==================合并的数据集========结束=======")
print('天气数据总记录数===》',df_aq.shape[0])
print('合并后总记录数===》',df_merge.shape[0])


# print("==================处理插值========开始=======")
# # 全字段插值方法
# def interpolation_all_columns():
#     pass
#
# print("==================处理插值========结束=======")
print("==================合并天气========开始=======")
meo_weather_data = pd.read_csv("assets/output/merge_aq_meo/merge_aq_meo.csv")
df_meo_weather = pd.DataFrame(meo_weather_data,index=None,columns=['stationId_aq','utc_time','weather'])
df_merge = pd.merge(df_merge,df_meo_weather,on=['stationId_aq','utc_time'],how='left')
print("==================合并天气========结束=======")
print("==================写入文件merge_aq_meo.csv========开始=======")
df_merge.to_csv('assets/output/merge_aq_meo/final_merge_aq_grid_meo2.csv',index=False)
print("==================写入文件merge_aq_meo.csv========结束=======")
