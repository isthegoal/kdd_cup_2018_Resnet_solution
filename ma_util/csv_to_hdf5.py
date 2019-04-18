import numpy as np
import pandas as pd
from datetime import datetime
import h5py
import time
from sklearn.preprocessing import MinMaxScaler
'''
代码将  标准化处理好的数据，按照大程序所需要的   h5文件中的内容形式，将 数据进行了 维度组合 和拼接，  并将日期数据进行格式转换， 分别放到三个列表中，之后保存h5文件中，、
等待模型的读取。  
'''


#我们需要使用新的标准化方式，  这里这个标准化方式太几把  麻烦了，还得写那么多东西
filename = '/home/fly/PycharmProjects/DeepST-KDD for_train/data/4-14_new_data/2017_data/final_merge_aq_grid_meo_with_weather4-17.h5'
read_csvfile=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD for_train/for_submit_data/new_data_5-5/final_merge_aq_grid_meo_deal_new.csv')
# ========================      对数据读取并标准化处理下（标准化和反标准化需要一个去适配）      ===========================#
scaler = MinMaxScaler(feature_range=(-1, 1))  # 数值限定到-1到1之间 看出来了，这是在同一个scaler下，将tranform转换后的数据，使用reverse还原回来，scaler有记忆的，所以在这里没用
#fit之前先挑选出全都是数值型的列     这里针对数值型进行按序列标准化之后再进行拼接，这里在使用fit_transform之后列的标题会丢掉，我又重加了进去。。。
scaler.fit(read_csvfile[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']])  # Compute the minimum and maximum to be used for later scaling 这是个fit过程， 这个过程会 计算以后用于缩放的平均值和标准差， 记忆下来
data1 = pd.DataFrame(scaler.fit_transform(read_csvfile[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]),columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3'])  # Fit to data, then transform it. 使用记忆的数据进行转换
data2 = read_csvfile[['stationId_aq', 'utc_time']]
frames = [data1, data2]  #在列上进行拼接。。。    还有很多链接方式的选择，左右、外链接等方式，全在书上，方式很多。。。
read_csvfile = pd.concat(frames, axis=1)



# ========================      插曲-借助进行反标准化 （这里很大的问题是没有天气信息，我随便给的天气，发现预测结果还不对，对着那）     ===========================#
# ok_filename = pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD/data/my_process/the_predict_data.csv')
#
# pred_array = pd.DataFrame(scaler.inverse_transform(ok_filename[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]),columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3'])  #使用记忆的数据反转换
# pred_array1 = ok_filename[['utc_time']]
# frames1 = [pred_array, pred_array1]
# read_csvfile1 = pd.concat(frames1, axis=1)
# #pred_array=pd.DataFrame(read_csvfile1,columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3','utc_time'])
# pred_array=pd.DataFrame(read_csvfile1)
# pred_array.to_csv('/home/fly/PycharmProjects/DeepST-KDD/data/my_process/the_predict_data.csv',index=False)
#



print('the standrad data:',read_csvfile)

# ========================      开始适配到h5文件中进行保存（没一个时刻点可在一起，可以看到组成了35*11的维度）      ===========================#
#print(read_csvfile['utc_time'])
big_list=[]
#list1=[[read_csvfile['temperature'][i],read_csvfile['pressure'][i],read_csvfile['humidity'][i],read_csvfile['wind_direction'][i]] for i in range(len(read_csvfile))]
list_big_data=[]
list_date=[]
list_35=[]
list_big_data_air=[]
list_date_air=[]
list_35_air=[]
list_pred_48_air=[]
list_big_pred_48_air=[]
#将所有数据  按时间点归于空气质量监测站 组合的函数。     把空气污染物信息那三个单独放到air文件中去。
for i in range(len(read_csvfile)):
    #list_small=[]
    list_35.append([read_csvfile['temperature'][i],read_csvfile['pressure'][i],read_csvfile['humidity'][i],read_csvfile['wind_direction'][i],read_csvfile['wind_speed/kph'][i],read_csvfile['weather'][i],read_csvfile['NO2'][i],read_csvfile['CO'][i],read_csvfile['SO2'][i],read_csvfile['PM2.5'][i],read_csvfile['PM10'][i],read_csvfile['O3'][i]])
    list_35_air.append([read_csvfile['PM2.5'][i],read_csvfile['PM10'][i],read_csvfile['O3'][i]])
    list_pred_48_air.append([read_csvfile['temperature'][i],read_csvfile['pressure'][i],read_csvfile['humidity'][i],read_csvfile['wind_direction'][i],read_csvfile['wind_speed/kph'][i],read_csvfile['weather'][i]])
    list_date.append(read_csvfile['utc_time'][i])
    if(i+1)% 35 == 0:
        list_big_data.append(list_35)
        list_big_data_air.append(list_35_air)
        list_big_pred_48_air.append(list_pred_48_air)
        list_35_air=[]
        list_35=[]
        list_pred_48_air=[]

list_big_data_np=np.array(list_big_data)
list_big_data_air=np.array(list_big_data_air)
list_big_pred_48_air=np.array(list_big_pred_48_air)
print(list_big_data_np.shape)
print(list_big_pred_48_air.shape)
#Python time strptime() 函数根据指定的格式把一个时间字符串解析为时间元组。  这里是将时间格式进行一次转换。。。。
#下面是从原始的时间格式转换成 需要的时间格式的方法   但是这里并不是所有的都是用上，而是要和之前的相对应（这就要保持自己一定要是每小时保持35个时间戳。）
new_list_date=[]
for t in range(len(list_date)):
    middle = datetime.strptime(list_date[t], '%Y-%m-%d %H:%M:%S')
    last=str(middle.strftime('%Y%m%d%H'))  #限定之后的真实转换   转成之后再变成字符串
    if (t + 1) % 35 == 0:
        new_list_date.append(last.encode())
#list_date = [time.strptime(str(t[:]), '%Y-%m-%d %H:%M:%S') for t in list_date]
print(new_list_date)
list_date_np=np.array(new_list_date)
print(list_date_np.shape)


######################现在开始讲  整理好的数据进行转换，加上标识转到h5文件中去
#filename='G:/Machine-learning/tensorflow-train-project(jupyter)/KDD-FreshAir/kDD 2018/Data/final_merge_aq_grid_meo_with_weather.h5'
f = h5py.File(filename,'w')
f['data']=list_big_data_np
f['data_air']=list_big_data_air
f['date']=list_date_np
f['pred_48air']=list_big_pred_48_air