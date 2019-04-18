import pandas as pd

#我们针对  标准化之前的数据删除以及加上几个特征，尝试下结果

#-------------------------------     尝试1： 去除影响性小的特征，加上三个特征， try1  ---------------------------------#
bejDf=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD for_train/for_submit_data/new_data_5-5/final_merge_aq_grid_meo_deal_new.csv')
bejDf.drop(labels='pressure', axis=1, inplace=True)
bejDf['utc_time'] = pd.to_datetime(bejDf['utc_time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
bejDf['hour'] = bejDf['utc_time'].dt.hour
bejDf['day'] = bejDf['utc_time'].dt.day
bejDf['month'] = bejDf['utc_time'].dt.month
bejDf['dayofweek'] = bejDf['utc_time'].dt.dayofweek

#统一对应转回去    不能这样，会出问题
#bejDf['utc_time']=str(bejDf['utc_time'])
print(bejDf)
bejDf.to_csv('/home/fly/PycharmProjects/DeepST-KDD for_train/for_submit_data/feature_engine_data/try1.csv')


