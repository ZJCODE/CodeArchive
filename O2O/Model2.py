# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:57 2016

@author: ZJun
"""

import pandas as pd
from sklearn.svm import SVC 
import numpy as np
from sklearn import metrics  
import random
from sklearn import ensemble


path1 = './Data/ccf_offline_stage1_test_revised.csv'
offline_test= pd.read_csv(path1,names= ['user', 'merchant', 'coupon', 'discount','distance', 'date_received'])

path2 = './Data/ccf_offline_stage1_train.csv'
offline_train = pd.read_csv(path2,names=['user', 'merchant', 'coupon', 'discount', 'distance', 'date_received', 'date'])

path3 = './Data/ccf_online_stage1_train.csv'
online_train =pd.read_csv(path3,names=['user', 'merchant','action', 'coupon', 'discount', 'date_received', 'date']) 


offline_train_data = pd.read_csv('./Data/offline_train_data.csv',parse_dates = ['date_received'])
online_train_data = pd.read_csv('./Data/online_train_data.csv',parse_dates = ['date_received'])
offline_test_data = pd.read_csv('./Data/offline_test_data.csv',parse_dates = ['date_received'])

def GetAction(online_train):
    action_data = online_train[['user','action']]
    action_dict = dict(action_data.pivot_table(values='action',index='user',aggfunc='mean'))
    return action_dict    
 
def AddAction(offline_test_data,offline_train_data,action_dict):    
    a_r = []
    for u in offline_test_data.user:
        try:
            a_r.append(action_dict[u])
        except:
            a_r.append(0.449)
    offline_test_data['a_r'] = a_r
    
    a_r = []
    for u in offline_train_data.user:
        try:
            a_r.append(action_dict[u])
        except:
            a_r.append(0.449)
    offline_train_data['a_r'] = a_r


def GetDeltaDict(online_train_data,offline_train_data):
    on = online_train_data[['user','delta']]
    off = offline_train_data[['user','delta']]
    data = pd.concat([on,off])
    data['delta'] = [15 if a == 'null' else int(a) for a in data.delta]
    delta_dict = dict(data.pivot_table(values='delta',index='user',aggfunc='mean'))
    return delta_dict

def AddDelta(offline_test_data,offline_train_data,delta_dict):    
    d_r = []
    for u in offline_test_data.user:
        try:
            d_r.append(delta_dict[u])
        except:
            d_r.append(14.11)
    offline_test_data['d_r'] = d_r
    
    d_r = []
    for u in offline_train_data.user:
        try:
            d_r.append(delta_dict[u])
        except:
            d_r.append(14.11)
    offline_train_data['d_r'] = d_r
    




def GetTrain(offline_train_data,coupon_id):
    data = offline_train_data[offline_train_data.coupon == coupon_id]
    data['weekday'] = [date.weekday() for date in data.date_received]
    X = data[['user','distance','weekday','d_r','u_r','a_r']].values
    y = data.tag.values
    return X,y
    
def GetTest(offline_test_data,offline_test,coupon_id):
    data = offline_test_data[offline_test_data.coupon == coupon_id]
    for_col = offline_test[offline_test.coupon == coupon_id][['user','coupon','date_received']]
    X_test = data[['user','distance','weekday','d_r','u_r','a_r']].values
    return X_test,for_col
    

def gbClassifierPred(X_train, X_test, y_train):
    import time
    t1 = time.time()
    t1 = time.time()
    params = {'n_estimators': 300, 'max_depth': 3, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = clf.predict_proba(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    return pred


offline_test_data = pd.read_csv('offline_test_data_V2.csv',parse_dates = ['date_received'])
offline_train_data = pd.read_csv('offline_train_data_V2.csv',parse_dates = ['date_received'])

coupon_list = list(set(offline_test_data.coupon))
c = coupon_list[0]
X,y = GetTrain(offline_train_data,c)
X_test,for_col = GetTest(offline_test_data,offline_test,c)
if len(set(y)) == 1:
    Prob = np.zeros(X_test.shape[0])
else:    
    Prob = [x[1] for x in gbClassifierPred(X, X_test, y)]

for_col['Prob'] = Prob
Result = for_col

for c in coupon_list[1:]:
    X,y = GetTrain(offline_train_data,c)
    X_test,for_col = GetTest(offline_test_data,offline_test,c)
    if len(set(y)) == 1 or len(y) == 0:
        Prob = np.zeros(X_test.shape[0])
    else:    
        Prob = [x[1] for x in gbClassifierPred(X, X_test, y)]
    for_col['Prob'] = Prob
    
    Result = pd.concat([Result,for_col])
    
Result = Result.sort_values(by='Prob',ascending=False)    
Result.to_csv('Predict.csv',index = False,header = False)

    
