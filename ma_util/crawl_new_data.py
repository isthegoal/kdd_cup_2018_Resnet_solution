import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
# 开始爬取初始数据
weather_url='https://biendata.com/competition/meteorology/bj/{start_time}/{end_time}/2k0d1d8'.format(start_time='2018-03-29-0',end_time='2018-04-16-17')
air_url='https://biendata.com/competition/airquality/{city}/{start_time}/{end_time}/2k0d1d8'.format(city='bj',start_time='2018-03-29-0',end_time='2018-04-16-17')
gird_url='https://biendata.com/competition/meteorology/{city}_grid/{start_time}/{end_time}/2k0d1d8'.format(city='bj',start_time='2018-03-29-0',end_time='2018-04-16-17')
#首先获取 18个气象监测站的天气信息，并把数据   切分和存储下来
weather_text=urllib.request.urlopen(weather_url).read()
split_weather_text=str(weather_text).split('\\r\\n')
split_weather_text=[i.split(',') for i in split_weather_text]
# for i in split_weather_text:
#     print('the weather :',i)
df_weather=pd.DataFrame(split_weather_text[1:len(split_weather_text)],columns=['id','station_id', 'time', 'weather', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed'])
df_weather.to_csv('/home/fly/PycharmProjects/DeepST-KDD/data/18weather_2018_3_31.csv')

#获取  35个监测站的检测信息，并将结果保存起来
air_text=urllib.request.urlopen(air_url).read()
split_air_text=str(air_text).split('\\r\\n')
split_air_text=[i.split(',') for i in split_air_text]
#for i in split_air_text:
#print('the air :',split_air_text[0],'',split_air_text[1])
df_weather=pd.DataFrame(split_air_text[1:len(split_air_text)],columns=['id','station_id', 'time', 'PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration'])
df_weather.to_csv('/home/fly/PycharmProjects/DeepST-KDD/data/air_2018_3_31.csv')


#获取  北京的网格数据，并将结果保存起来
gird_text=urllib.request.urlopen(gird_url).read()
split_gird_text=str(gird_text).split('\\r\\n')
split_gird_text=[i.split(',') for i in split_gird_text]
#for i in split_air_text:
print('the gird :',split_gird_text[0],'',split_gird_text[1])
df_gird=pd.DataFrame(split_gird_text[1:len(split_gird_text)],columns=['id','station_id', 'time', 'weather', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed'])
df_gird.to_csv('/home/fly/PycharmProjects/DeepST-KDD/data/gird_2018_3_31.csv')



