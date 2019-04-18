import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
data = pd.read_csv("/home/fly/PycharmProjects/DeepST-KDD/ma_util/data_merge_process/data/beijing_17_18_aq.csv",parse_dates=True,index_col='utc_time')

'''

这个函数是  展示出 35个站点上空气质量情况
写的很强， 可以从图像上观察出明显的趋势性，所以说趋势性的数据还是需要考虑的。。。

可以学习学习，不要随便画出来，图太多了。
'''
#times = pd.date_range(start = '2017/1/1 14:00:00',end = '2018/1/31 15:00:00',freq ='60min')
#mydates = data['utc_time']
df = pd.DataFrame(data,None,columns=['stationId','PM2.5','PM10','O3'])
#df.set_index(['utc_time'],inplace = True)
df = df.dropna(axis=0,how='all')   #删除所有为NULL的数据， 这里all方式默认是删除全为NULL的行，axis指定是删除行上数据
# df1 = df[df['stationId']=='wanliu_aq']
gdf = df.groupby(data['stationId'])  #以每个站点作为  一个分组去  画图
#df.groupby('stationId')
#df.plot()
fig = plt.figure()
plotNum = len(gdf.count())
i = 1
for name,group in --gdf:

    print(name)
    ax = fig.add_subplot(plotNum, 1, plotNum)
    group.plot()
    plt.title(name)
    i +=1
plt.show()
    #print(group)
#print(plotNum)
#print(gdf.count())
#plt.show()