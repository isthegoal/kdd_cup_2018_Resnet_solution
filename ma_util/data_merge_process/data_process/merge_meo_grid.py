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
from ma_util.data_merge_process.data_process.tools import *

'''
处理质量站的数据
将网格数据距离最近的质量站数据进行经纬度合并替换

对于其作用最好的表示就是看其保存的数据为
 stationName  longitude  latitude             utc_time  temperature  pressure  humidity  wind_direction  wind_speed/kph  
可以看出是将 站点数据和网格数据进行合并。。。
'''

def __search_aq_grid_dict(aq_grid_dict,aq_longitude,aq_latitude):
    for aq_name_grid, gridTuple in aq_grid_dict.items():
        target_longitude = gridTuple[0]
        target_latitude = gridTuple[1]
        if aq_longitude == target_longitude and aq_latitude == target_latitude:
            return True
    return False

def ProcessAqData():
    aq_grid_dict = {}
    aq_station_data = pd.read_csv("assets/data/Beijing_AirQuality_Stations_cn_data.csv")
    aq_grid_data = pd.read_csv("assets/data/Beijing_historical_meo_grid.csv")
    df_aq_grid = pd.DataFrame(aq_grid_data,index=None,columns=['stationName','longitude','latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed/kph'])
    df_aq_grid_drop = df_aq_grid.drop_duplicates(['longitude','latitude'])
    for aq_index in aq_station_data.index:
        aq_name = aq_station_data.loc[aq_index].values[0]
        aq_longitude = aq_station_data.loc[aq_index].values[1]
        aq_latitude = aq_station_data.loc[aq_index].values[2]

        cur_min_dis = None
        cur_min_tuple = None
        for index in df_aq_grid_drop.index:
            longitude = df_aq_grid_drop.loc[index].values[1]
            latitude = df_aq_grid_drop.loc[index].values[2]
            dis = Distance1(latitude,longitude,aq_latitude,aq_longitude)
            if cur_min_dis is None:
                cur_min_dis = dis
                cur_min_tuple = (longitude, latitude)
            else:
                if dis <= cur_min_dis:
                    cur_min_dis = dis
                    cur_min_tuple = (longitude,latitude)

        aq_grid_dict[aq_name] = cur_min_tuple

    contact_result_grid = []
    for k,v in aq_grid_dict.items():
        aq_name = k
        aq_longitude = v[0]
        aq_latitude  = v[1]
        df_aq_grid_temp = df_aq_grid[(df_aq_grid.longitude == aq_longitude) & (df_aq_grid.latitude == aq_latitude)]
        df_aq_grid_temp.stationName[(df_aq_grid.longitude == aq_longitude) & (df_aq_grid.latitude == aq_latitude)] = aq_name
        contact_result_grid.append(df_aq_grid_temp)

    result_data = pd.concat(contact_result_grid)

    print('result_data:',result_data)
    result_data.to_csv('assets/output/merge_aq_meo/merge_grid_meo.csv',index=False)
    # del_row_index_list = []
    # for aq_index in df_aq_grid.index:
    #     aq_longitude = df_aq_grid.loc[aq_index].values[1]
    #     aq_latitude = df_aq_grid.loc[aq_index].values[2]
    #     search_result = __search_aq_grid_dict(aq_grid_dict, aq_longitude, aq_latitude)
    #     if not search_result:
    #         del_row_index_list.append(aq_index)
    #
    # df_aq_grid = df_aq_grid.drop(del_row_index_list)
    # for k,v in aq_grid_dict.items():
    #     print(k,v)


ProcessAqData()