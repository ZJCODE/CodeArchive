# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 22:03:16 2016

@author: ZJun
"""

import pandas as pd
import numpy as np



def LoadData():
    path1 = './Data/ccf_offline_stage1_test_revised.csv'
    offline_test= pd.read_csv(path1,names= ['user', 'merchant', 'coupon', 'discount','distance', 'date_received'])

    path2 = './Data/ccf_offline_stage1_train.csv'
    offline_train = pd.read_csv(path2,names=['user', 'merchant', 'coupon', 'discount', 'distance', 'date_received', 'date'])
    
    path3 = './Data/ccf_online_stage1_train.csv'
    online_train =pd.read_csv(path3,names=['user', 'merchant','action', 'coupon', 'discount', 'date_received', 'date']) 
    
    return offline_test,offline_train,online_train


'''
len(set(offline_test.user))
Out[83]: 76309

len(set(offline_test.user) & (set(online_train.user) | set(offline_train.user)))
Out[84]: 76308

len(set(offline_test.merchant) & (set(online_train.merchant) | set(offline_train.merchant)))
Out[86]: 1558

len(set(offline_test.merchant))
Out[87]: 1559

'''


def DealWith_offline_train(offline_train):
     
    date_received = []
    for d in offline_train.date_received:
        if d == 'null':
            date_received.append(d)
        else:
            date_received.append(pd.to_datetime(d))
    offline_train['date_received'] = date_received
    
    date = []
    for d in offline_train.date:
        if d == 'null':
            date.append(d)
        else:
            date.append(pd.to_datetime(d))
    offline_train['date'] = date
        
    delta = []
    
    for t1,t2 in zip(date,date_received):
        if t1 == 'null' or t2 == 'null':
            delta.append('null')
        else:
            delta.append((t1-t2).days)
    offline_train['delta'] = delta
    
    tag = []
    for date , coupon ,delta in zip(offline_train.date,offline_train.coupon,offline_train.delta):
        if date == 'null' and coupon != 'null':
            tag.append(0)
        elif date != 'null' and coupon != 'null' and delta < 15:
            tag.append(1)
        else:
            tag.append(np.nan)
    offline_train['tag'] = tag
    
    discount = []  # need to process more
    for d in offline_train.discount:
        if ':' in d:
            n1,n2 = [int(n) for n in d.split(':')]
            discount.append((n1-n2)*1.0/n1)
        elif '.' in d:
            discount.append(float(d))
        else:
            discount.append(np.nan)
    offline_train['discount'] = discount
    

    coupon = []
    for c in offline_train.coupon:
        if c == 'null':
            coupon.append(np.nan)
        else:
            coupon.append(int(c))
    offline_train['coupon'] = coupon
    
    
    distance = []
    for d in offline_train.distance:
        if d == 'null' or np.isnan(d):
            distance.append(3)
        else:
            distance.append(int(d))
    offline_train['distance'] = distance
        
    
    offline_train = offline_train.dropna()
    
    offline_train.index = range(len(offline_train))
    offline_train.to_csv('offline_train.csv',index=False)        
        
    return offline_train



def DealWith_online_train(online_train):
    
    date_received = []
    i=0
    for d in online_train.date_received:
        i = i+1
        print '===deal with' + str(i) +'th==='
        if d == 'null':
            date_received.append(d)
        else:
            date_received.append(pd.to_datetime(d))
    online_train['date_received'] = date_received
    
    date = []
    i=0
    for d in online_train.date:
        i = i+1
        print '===deal with' + str(i) +'th==='
        if d == 'null':
            date.append(d)
        else:
            date.append(pd.to_datetime(d))
    online_train['date'] = date
        
    delta = []
    
    for t1,t2 in zip(date,date_received):
        if t1 == 'null' or t2 == 'null':
            delta.append('null')
        else:
            delta.append((t1-t2).days)
    online_train['delta'] = delta
    
    tag = []
    for date , coupon ,delta in zip(online_train.date,online_train.coupon,online_train.delta):
        if date == 'null' and coupon != 'null':
            tag.append(0)
        elif date != 'null' and coupon != 'null' and delta < 15:
            tag.append(1)
        else:
            tag.append(np.nan)
    online_train['tag'] = tag
    
    discount = []
    for d in online_train.discount:
        if ':' in d:
            n1,n2 = [int(n) for n in d.split(':')]
            discount.append((n1-n2)*1.0/n1)
        elif '.' in d:
            discount.append(float(d))
        else:
            discount.append(np.nan)
    online_train['discount'] = discount
    
    coupon = []
    for c in online_train.coupon:
        if c == 'null':
            coupon.append(np.nan)
        elif c== 'fixed':
            coupon.append(np.nan)
        else:
            coupon.append(int(c))
    online_train['coupon'] = coupon    
    
    online_train = online_train.dropna()
    
    online_train.index = range(len(online_train))
    online_train.to_csv('online_train.csv',index=False)
    return online_train    



def DealWith_offline_test(offline_test):
    
    date_received = []
    i=0
    for d in offline_test.date_received:
        i = i+1
        print '===deal with' + str(i) +'th==='
        if d == 'null':
            date_received.append(d)
        else:
            date_received.append(pd.to_datetime(str(d)))
    offline_test['date_received'] = date_received
    
    discount = []
    for d in offline_test.discount:
        if ':' in d:
            n1,n2 = [int(n) for n in d.split(':')]
            discount.append((n1-n2)*1.0/n1)
        elif '.' in d:
            discount.append(float(d))
        else:
            discount.append(np.nan)
    offline_test['discount'] = discount
    
    distance = []
    for d in offline_test.distance:
        if d == 'null' or np.isnan(d):
            distance.append(3)
        else:
            distance.append(int(d))
    offline_test['distance'] = distance
    
    offline_test['weekday'] = [date.weekday() for date in offline_test.date_received]
    
    offline_test.index = range(len(offline_test))
    offline_test.to_csv('offline_test.csv',index=False)    
    
    return offline_test


def GetUserRatio(offline_train,online_train):
    off = offline_train[['user','tag']]
    on = online_train[['user','tag']]
    users_tag = pd.concat([off,on])
    user_dict = dict(users_tag.pivot_table(values='tag',index='user',aggfunc='mean'))
    return user_dict

def GetMerchantRatio(offline_train,online_train):
    off = offline_train[['merchant','tag']]
    on = online_train[['merchant','tag']]
    merchant_tag = pd.concat([off,on])
    merchant_dict = dict(merchant_tag.pivot_table(values='tag',index='merchant',aggfunc='mean'))
    return merchant_dict    

def GetCouponRatio(offline_train,online_train):
    off = offline_train[['coupon','tag']]
    on = online_train[['coupon','tag']]
    coupon_tag = pd.concat([off,on])
    coupon_dict = dict(coupon_tag.pivot_table(values='tag',index='coupon',aggfunc='mean'))
    return coupon_dict    


    

    

'''
np.mean(user_dict.values())
Out[106]: 0.071026262357253928

np.mean(merchant_dict.values())
Out[107]: 0.20277610788480382

np.mean(coupon_dict.values())
Out[108]: 0.23885948335922028

np.mean(action_dict.values())
Out[37]: 0.44930947352803313


np.mean(delta_dict.values())
Out[70]: 14.110268929871907

'''


def AddRatio(Data,user_dict,merchant_dict,coupon_dict,action_dict):
    u_r=[]
    for u in Data.user:
        try:
            u_r.append(user_dict[u])
        except:
            u_r.append(0.07)
    Data['u_r'] = u_r
    
    m_r = []
    for m in Data.merchant:
        try:
            m_r.append(merchant_dict[m])
        except:
            m_r.append(0.202)
    Data['m_r'] = m_r
    
    c_r = []
    for c in Data.coupon:
        try:
            c_r.append(coupon_dict[c])
        except:
            c_r.append(0.238)
    Data['c_r'] = c_r
    
    
    return Data

def main():
    
    offline_test,offline_train,online_train = LoadData()
    
    offline_train = DealWith_offline_train(offline_train)    
    online_train = DealWith_online_train(online_train)
    offline_test = DealWith_offline_test(offline_test)
    
    user_dict = GetUserRatio(offline_train,online_train)
    merchant_dict = GetMerchantRatio(offline_train,online_train)
    coupon_dict = GetCouponRatio(offline_train,online_train)
    
    offline_train_data = AddRatio(offline_train,user_dict,merchant_dict,coupon_dict)
    online_train_data = AddRatio(online_train,user_dict,merchant_dict,coupon_dict)
    offline_test_data = AddRatio(offline_test,user_dict,merchant_dict,coupon_dict)
    
    return offline_train_data,online_train_data,offline_test_data
    



'''
offline_test = pd.read_csv('offline_test.csv',parse_dates = ['date_received'])
offline_train =pd.read_csv('offline_train.csv',parse_dates = ['date_received'])
online_train = pd.read_csv('online_train.csv',parse_dates = ['date_received'])
'''