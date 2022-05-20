import numpy as np
import pandas as pd
data_pd = pd.read_csv("douban_music.tsv",header=0,sep='\t')
# data_pd.head(2)
# data_pd['date'] = pd.to_datetime(data_pd['Timestamp'],unit='s')
# data_pd.head(2)
# data_pd['date'].min()
# data_pd['date'].max()
# data_pd.shape
# data_pd['year'] = data_pd['date'].dt.year
# data_pd.groupby(['year']).agg({'Rating':'count'})
# data_pd = data_pd[data_pd['year']>=2010]
data_pd = data_pd.drop_duplicates(subset=['UserId','ItemId'],ignore_index=True)
# data_pd.shape
data_pd.groupby('Rating').agg({'ItemId':'count'})

data_pd=data_pd[data_pd.Rating!=-1]
data_pd.to_csv('DoubanMusic.csv',index=False,sep='\t')


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


# time_min = data['Timestamp'].min()
# time_max = data['Timestamp'].max()
# slot_gap = (time_max - time_min) /10
# data['time_slot'] = data["Timestamp"].apply(lambda x: int(min(int((x-time_min))//slot_gap,9)))
# data['time_slot'] = data[['time_slot']].astype(np.int)
# timestamp = time_min + slot_gap
# import time

# #转换成localtime
# time_local = time.localtime(timestamp)
# dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
# dt
# data.head(4)
# train_slots = [0 ,1, 2, 3, 4, 5, 6,7,8]
# test_slots = [9]
# data_train = data[data['time_slot'].isin(train_slots)]
# data_test = data[data['time_slot'].isin(test_slots)]
# print("train:",data_train.shape[0],'test:',data_test.shape[0])
# user_in_train = data_train['UserId'].unique()
# item_in_train = data_train['ItemId'].unique()

# filter new user/item in train
# data_test = data_test[data_test['UserId'].isin(user_in_train)]
# print("user not include in user_items_test:",data_test.shape)
# data_test = data_test[data_test['ItemId'].isin(item_in_train)]
# print("train:",data_train.shape[0],'not-new test:',data_test.shape[0])

# # filter repeat
# data_train = data_train.drop_duplicates(subset=['UserId','ItemId'],keep='first')
# data_test = data_test.drop_duplicates(subset=['UserId','ItemId'],keep='first')
# print("not repeat train:",data_train.shape[0],'not-repeat test:',data_test.shape[0])

# user = data_train['UserId'].unique()
# item= data_train['ItemId'].unique()
# user_to_id = dict(zip(list(user),list(np.arange(user.shape[0]))))
# item_to_id = dict(zip(list(item),list(range(item.shape[0]))))
# print("user num:",user.shape)
# print("item num:", item.shape)
# data_train['uid'] = data_train['UserId'].map(user_to_id)
# data_train['iid'] = data_train['ItemId'].map(item_to_id)

# data_test['uid'] = data_test['UserId'].map(user_to_id)
# data_test['iid'] = data_test['ItemId'].map(item_to_id)

# data_test.head(2)
# continue
# data_train = data_train[['uid','iid','time_slot','Rating']]
# data_test = data_test[['uid','iid','time_slot','Rating']]
# #
# columns = ['uid','iid','time_slot','click']
# data_train.columns = columns
# data_test.columns = columns
# data_test.head(2)
# # real time
 
# # split testing and valuation
# data_test['uid'].unique().shape
 
# test_unique_user = data_test['uid'].unique()
# N_ = test_unique_user.shape[0]
# np.random.seed(2020)
# np.random.shuffle(test_unique_user)
# split_idx  = int(N_*0.7)
# test_real_user = test_unique_user[:split_idx]
# valid_real_user = test_unique_user[split_idx:]
# print("tot user in the last stage:",N_,"real test user:",test_real_user.shape[0],"real valid user:",valid_real_user.shape[0])
 
# data_real_test = data_test[data_test['uid'].isin(test_real_user)]
# data_real_valid = data_test[data_test['uid'].isin(valid_real_user)]
# print("tot itr:",data_test.shape,"real test:",data_real_test.shape,"real valid:",data_real_valid.shape)
# saving
# import os
# path_folder = "./douban_moive/"
# if not os.path.exists(path_folder):
#     os.mkdir(path_folder)
# user_items_test = data_real_test.sort_values(by='uid',ignore_index=True)
# print(user_items_test.head(2))
# test_itr = user_items_test.values[:,0:2]
# print(test_itr.shape)
# with open('./douban_moive/test_real.txt','w') as f:
#     u_pre = test_itr[0,0]
#     k = 0
#     for x in test_itr:
#         u = x[0]
#         i = x[1]
#         if u !=u_pre or k==0:
#             u_pre = u
#             if k>0:
#                 f.write('\n')
#             f.write(str(u))
#             k = 1
#         f.write(' '+str(i))
# user_items_test = data_real_valid.sort_values(by='uid',ignore_index=True)
# print(user_items_test.head(2))
# test_itr = user_items_test.values[:,0:2]
# print(test_itr.shape)
# with open('./douban_moive/valid_real.txt','w') as f:
#     u_pre = test_itr[0,0]
#     k = 0
#     for x in test_itr:
#         u = x[0]
#         i = x[1]
#         if u !=u_pre or k==0:
#             u_pre = u
#             if k>0:
#                 f.write('\n')
#             f.write(str(u))
#             k = 1
#         f.write(' '+str(i))
 
# user_items_test = data_train.sort_values(by='uid',ignore_index=True)
# print(user_items_test.head(2))
# test_itr = user_items_test.values[:,0:2]
# print(test_itr.shape)
# with open('./douban_moive/train.txt','w') as f:
#     u_pre = test_itr[0,0]
#     k = 0
#     for x in test_itr:
#         u = x[0]
#         i = x[1]
#         if u !=u_pre or k==0:
#             u_pre = u
#             if k>0:
#                 f.write('\n')
#             f.write(str(u))
#             k = 1
#         f.write(' '+str(i))
 
# data_train.to_csv("./douban_moive/train_with_time.txt",index=False,header=False,sep=' ')
# data_real_valid.to_csv("./douban_moive/valid_with_time.txt",index=False,header=False,sep=' ')
# data_real_test.to_csv("./douban_moive/test_with_time.txt",index=False,header=False,sep=' ')
# data_train['iid'].max()
# for slot_id in train_slots:
#     slot_data = data_train[data_train['time_slot'].isin([slot_id])]
#     slot_data  = slot_data.sort_values(by=['iid'],ignore_index=True)
#     slot_data_np = slot_data[['iid','uid']].values[:,0:2]
#     print(slot_data.head(2))
#     print(slot_data[['iid','uid']].head(2))
#     print(slot_data.shape)
#     with open("./douban_moive/t_"+str(slot_id)+".txt",'w') as f:
#         i_pre = slot_data_np[0,0]
#         k = 0
#         for x in slot_data_np:
#             i_ = x[0]
#             u_ = x[1]
#             if i_ != i_pre or k == 0:
#                 i_pre = i_
#                 if k>0:
#                     f.write('\n')
#                 f.write(str(i_))
#                 k = 1
#             f.write(" " + str(u_))
 
# slot_data = data_test
# slot_data  = slot_data.sort_values(by=['iid'],ignore_index=True)
# slot_data_np = slot_data[['iid','uid']].values[:,0:2]
# print(slot_data.head(2))
# print(slot_data[['iid','uid']].head(2))
# print(slot_data.shape)
# with open("./douban_moive/t_"+str(9)+".txt",'w') as f:
#     i_pre = slot_data_np[0,0]
#     k = 0
#     for x in slot_data_np:
#         i_ = x[0]
#         u_ = x[1]
#         if i_ != i_pre or k == 0:
#             i_pre = i_
#             if k>0:
#                 f.write('\n')
#             f.write(str(i_))
#             k = 1
#         f.write(" " + str(u_))