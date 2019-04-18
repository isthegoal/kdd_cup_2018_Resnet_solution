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


'''
计算经纬度之间的距离方法一
@Lat  is latitude 维度
@Lng is longitude 经度
'''

aq_grid_data = pd.read_csv("assets/data/Beijing_historical_meo_grid.csv")
df_aq_grid = pd.DataFrame(aq_grid_data)
df_aq_grid = df_aq_grid.drop_duplicates(['longitude','latitude'])#drop_duplicates（）DataFrame格式的数据，去除特定列下面的重复行。返回DataFrame格式的数据
for index in aq_grid_data.index:
    print(aq_grid_data.loc[index].values[0])
#print(len(df_aq_grid))