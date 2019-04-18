import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
data = pd.read_csv("/home/fly/PycharmProjects/DeepST-KDD/ma_util/data_merge_process/data/beijing_17_18_meo.csv",parse_dates=True,index_col='utc_time')


#times = pd.date_range(start = '2017/1/1 14:00:00',end = '2018/1/31 15:00:00',freq ='60min')
#mydates = data['utc_time']
df = pd.DataFrame(data,None)
#df.set_index(['utc_time'],inplace = True)
#df = df.dropna(axis=0,how='all')
df.plot()

print(df)
plt.show()