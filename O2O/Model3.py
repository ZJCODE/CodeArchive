# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:40:27 2016

@author: ZJun
"""

import pandas as pd
from sklearn.svm import SVC 
import numpy as np
from sklearn import metrics  
import random
from sklearn import ensemble


online_train_data = pd.read_csv('./Data/online_train_data.csv',parse_dates = ['date_received'])

offline_train_data = pd.read_csv('./Data/offline_train_data_V2.csv',parse_dates = ['date_received'])
offline_test_data = pd.read_csv('./Data/offline_test_data_V2.csv',parse_dates = ['date_received'])


def GetDeltaDict(online_train_data,offline_train_data):
    on = online_train_data[['user','delta']]
    off = offline_train_data[['user','delta']]
    data = pd.concat([on,off])
    data['delta'] = [15 if a == 'null' else int(a) for a in data.delta]
    delta_dict = dict(data.pivot_table(values='delta',index='user',aggfunc='mean'))
    return delta_dict
    
delta_dict = GetDeltaDict(online_train_data,offline_train_data)

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

print 'data prepared'

def GetTest(offline_test_data):
    return offline_test_data[['user','merchant','coupon','discount','distance','weekday','u_r','m_r','c_r','a_r','d_r']].values

def GetTrain(offline_train_data,way = 'add'):
    if way == 'add':
        offline_train_data['weekday'] = [date.weekday() for date in offline_train_data.date_received]
        offline_train_data_1 = offline_train_data[offline_train_data.tag==1]
        for i in range(10):
            offline_train_data = pd.concat([offline_train_data,offline_train_data_1])
    elif way == 'drop':
        offline_train_data['weekday'] = [date.weekday() for date in offline_train_data.date_received]
        offline_train_data_1 = offline_train_data[offline_train_data.tag==1]
        offline_train_data_0 = offline_train_data[offline_train_data.tag==0]
        Index = range(len(offline_train_data_0))
        offline_train_data_0.index = Index
        Sample_Index = random.sample(Index,len(offline_train_data_1)*8)
        sample_offline_train_data_0 = offline_train_data_0.ix[Sample_Index]
        offline_train_data = pd.concat([offline_train_data_1,sample_offline_train_data_0])
    else:
        print 'Wrong Input'
    X = offline_train_data[['user','merchant','coupon','discount','distance','weekday','u_r','m_r','c_r','a_r','d_r']].values
    #X = offline_train_data[['discount','distance','weekday','u_r','m_r','c_r']].values
    y = offline_train_data.tag.values
    return X,y
    
X,y = GetTrain(offline_train_data,way='add')
X_Pred = GetTest(offline_test_data)

print 'get X,y'

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


def svmClassifier(X_train,X_test,y_train,y_test):
    import time
    t1 = time.time()
    svclf = SVC(kernel='linear')#default with 'rbf'  
    svclf.fit(X_train,np.array(y_train))  
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = svclf.predict(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    print metrics.auc(fpr, tpr)


def rfClassifier(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    import time
    t1 = time.time()
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train,np.array(y_train))  
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = rfc.predict(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    print metrics.auc(fpr, tpr)


def gbClassifier(X_train, X_test, y_train,y_test):
    import time 
    t1 = time.time()
    params = {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
              
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = clf.predict(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    print metrics.auc(fpr, tpr)
    
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


def rfClassifierPred(X_train,X_test,y_train):
    from sklearn.ensemble import RandomForestClassifier
    import time
    t1 = time.time()
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train,np.array(y_train))  
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = rfc.predict_proba(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    return pred

def svmClassifierPred(X_train,X_test,y_train):
    import time
    t1 = time.time()
    svclf = SVC(kernel='linear')#default with 'rbf'  
    svclf.fit(X_train,np.array(y_train))  
    t2 = time.time()
    print '========Model Fitted========== Cost : '+str(t2-t1) + ' Seconds'
    pred = svclf.predict_proba(X_test)
    t3 = time.time()
    print '========Predict Finished====== Cost : '+str(t3-t2) + ' Seconds'
    return pred


Pro = gbClassifierPred(X,X_Pred,y)
#Pro = rfClassifierPred(X,X_Pred,y)
#Pro = svmClassifierPred(X,X_Pred,y)


Prob = [x[1] for x in Pro]

path1 = './Data/ccf_offline_stage1_test_revised.csv'
offline_test= pd.read_csv(path1,names= ['user', 'merchant', 'coupon', 'discount','distance', 'date_received'])
Predict = pd.DataFrame({'user':offline_test.user,'coupon':offline_test.coupon,'date_received':offline_test.date_received,'Prob':Prob})
P = Predict[['user','coupon','date_received','Prob']]
P.to_csv('Predict.csv',index = False,header = False)


'''

Pro = [ur*mr*cr*ar for ur,mr,cr,ar in zip(offline_test_data.u_r,offline_test_data.m_r,offline_test_data.c_r,offline_test_data.a_r)]
offline_test_data['Pro'] = Pro

offline_test_data['date_received'] = offline_test.date_received
offline_test_data =offline_test_data.sort(columns=['Pro','u_r','m_r','c_r'],ascending = False)
Prob = np.arange(len(offline_test_data))
Prob =sorted(Prob,reverse=True)
Prob = np.array(Prob)*1.0 / Prob[0]
Predict = pd.DataFrame({'user':offline_test_data.user,'coupon':offline_test_data.coupon,'date_received':offline_test_data.date_received,'Prob':Prob})
P = Predict[['user','coupon','date_received','Prob']]
P.to_csv('Predict.csv',index = False,header = False)
'''