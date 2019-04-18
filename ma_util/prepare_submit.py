#----------------  合并将自己还原的结果改成要处理的数据  -----------------#

import pandas as pd
from math import*
import matplotlib.pyplot as plt


# 读入数据文件        会产生自己需要的  要提交的那种文件结果
aq_data = pd.read_csv("data/final_merge_aq_grid_meo_with_weather4-14.csv")
predict_data = pd.read_csv("data/the_predict_data.csv")
df_aq = pd.DataFrame(aq_data,index=None,columns=['stationId_aq'])
df_aq.rename(columns={'stationId_aq':'test_id'},inplace=True)
df_aq_2100 = df_aq.loc[0:2099,:]
print(df_aq_2100)
for index,row in df_aq_2100.iterrows():
    row['test_id'] = row['test_id']+"#"+str(index)
#print(len(df_aq_2101))
df_predict = pd.DataFrame(predict_data,index=None,columns=['PM2.5','PM10','O3'])
df_merge = pd.concat([df_aq_2100,df_predict],axis=1)
df_merge.to_csv('output/merge_aq_meo/merge_predict.csv',index=False)