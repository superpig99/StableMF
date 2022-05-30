import numpy as np
import pandas as pd
data_pd = pd.read_csv("douban_music.tsv",header=0,sep='\t')
data_pd = data_pd.drop_duplicates(subset=['UserId','ItemId'],ignore_index=True)
data_pd.groupby('Rating').agg({'ItemId':'count'})

data_pd=data_pd[data_pd.Rating!=-1]

def filter_g_k_one(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):
    item_group = data.groupby(i_name).agg({y_name:'count'})
    item_g10 = item_group[item_group[y_name]>=k].index
    data_new = data[data[i_name].isin(item_g10)]
    user_group = data_new.groupby(u_name).agg({y_name:'count'})
    user_g10 = user_group[user_group[y_name]>=k].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new

def filter_tot(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):
    data_new=data
    while True:
        data_new = filter_g_k_one(data_new,k=k,u_name=u_name,i_name=i_name,y_name=y_name)
        m1 = data_new.groupby(i_name).agg({y_name:'count'})
        m2 = data_new.groupby(u_name).agg({y_name:'count'})
        num1 = m1[y_name].min()
        num2 = m2[y_name].min()
        print('item min:',num1,'user min:',num2)
        if num1>=k and num2>=k:
            break
    return data_new

data = filter_tot(data_pd,k=10,u_name='UserId',i_name='ItemId',y_name='Rating')
data.shape
data['UserId'].unique().shape
data['ItemId'].unique().shape

data.to_csv('DoubanMusic.csv',index=False,sep='\t')
