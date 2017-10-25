#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:18:08 2017

@author: zhangjun
"""
from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter
from get_ks import ks,print_ks,plot_ks
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn.metrics import roc_auc_score
import gc


def grid_search(param_grid):
    #from itertools import product
    items = sorted(param_grid.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params

def balance_data(X_train,y_train):
    data1 = X_train[y_train == 0]
    data2 = X_train[y_train == 1]
    data1_len = len(data1)
    data2_len = len(data2)
    if data1_len > data2_len:
        data_still = data1
        data_up_sample = data2
        data_up_sample_len = len(data_up_sample)
        n = int(data1_len/data2_len)
        y_train_balance = np.zeros(len(data_still))
        tag = 0
    else:
        data_still = data2
        data_up_sample = data1
        data_up_sample_len = len(data_up_sample)
        n = int(data2_len/data1_len)
        y_train_balance = np.ones(len(data_still))
        tag = 1
    X_train_balance = data_still
    for i in range(n):
        X_train_balance  = np.vstack([X_train_balance,data_up_sample])
        if tag == 0:
            y_train_balance = np.hstack([y_train_balance,np.ones(data_up_sample_len)])
        else:
            y_train_balance = np.hstack([y_train_balance,np.zeros(data_up_sample_len)])
    if isinstance(X_train,pd.core.frame.DataFrame):
        X_train_balance = pd.DataFrame(X_train_balance,columns = X_train.columns)
    return X_train_balance,y_train_balance

def deal_with_error(X_train):
    for feature in X_train.columns:
        up = X_train[feature].quantile(0.99)
        down = X_train[feature].quantile(0.01)
        X_train[feature].values[X_train[feature]>up] = up
        X_train[feature].values[X_train[feature]<down] = down
    return X_train

def fill_nan(X_train,y_train,way):
    fill_nan_val = []
    for feature in X_train.columns:
        index_null = pd.isnull(X_train[feature])
        if index_null.sum()>0:
            
            if way == 'dis':
                index_null = pd.isnull(X_train[feature])
                label_miss = y_train[index_null == True]
                miss_overdue_ratio = sum(label_miss) * 100 / (float(len(label_miss))+10e-8)
                ks_info = ks(y_train[index_null == False], X_train[feature][index_null == False], 20)
                delta_list = abs(ks_info['overdue_ratio'] - miss_overdue_ratio)
                span = ks_info['span_list'][delta_list.argmin()]
                try:
                    val1 = float(span.strip().split(',')[0].split('(')[1])
                except:
                    val1 = float(span.strip().split(',')[0].split('[')[1])
                val2 = float(span.strip().split(',')[1].split(']')[0])
                val = (val1 + val2) / 2.0

            elif way == 'avg':
                val = X_train[feature].mean()
            elif way == 'mid':
                val = X_train[feature].median()
            elif isinstance(way,int) or isinstance(way,float):
                val = way
            else:
                print 'error input , try again'
                return None,None
        else:
            val = None  
            
        X_train[feature] = X_train[feature].fillna(value = val)
        fill_nan_val.append(val)
        
    fill_nan_dict = dict(zip(X_train.columns,fill_nan_val))
    
    return X_train,fill_nan_dict


# RF

def RF_Fit(X_train,y_train,pars):
    rf = RandomForestRegressor(**pars)
    rf.fit(X_train,y_train)
    return rf

def RF_Predict(rf,X_test):
    pred = rf.predict_proba(X_test)[:,1]
    return pred

# Lr

def LR_Fit(X_train,y_train,pars):
    lr = LogisticRegression(**pars)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    lr.fit(X_train,y_train)
    return lr,scaler

def LR_Predict(lr,scaler,X_test):
    X_test = scaler.transform(X_test)
    pred = lr.predict_proba(X_test)[:,1]
    return pred

# GBDT

def GBDT_Fit(X_train,y_train,pars):
    gbdt = GradientBoostingClassifier(**pars)
    gbdt.fit(X_train,y_train)
    return gbdt

def GBDT_Predict(gbdt,X_test):
    pred = gbdt.predict_proba(X_test)[:,1]
    return pred

# XGBoost

def XGBoost_Fit(X_train,y_train,pars,num_round,X_val=None,y_val=None):
    #import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    try:
        _ = len(X_val)    
        dval_watch = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'),((dval_watch, 'dval_watch'))]
        xgb_model = xgb.train(pars, dtrain, verbose_eval=1, evals=watchlist,num_boost_round = num_round,early_stopping_rounds = 15)
    except:
        watchlist = [(dtrain, 'train')]
        xgb_model = xgb.train(pars, dtrain, verbose_eval=1,evals=watchlist,num_boost_round = num_round)
    return xgb_model

def XGBoost_Predict(xgb_model,X_test):
    dtest = xgb.DMatrix(X_test)
    pred = xgb_model.predict(dtest)
    return  pred

# GBDT_LR

def GBDTLR_Fit(X_train,y_train,pars):
    #from sklearn.ensemble import GradientBoostingClassifier
    #from sklearn.preprocessing import OneHotEncoder
    gbdt = GradientBoostingClassifier(**pars)
    gbdt.fit(X_train,y_train)
    model_onehot = OneHotEncoder()
    model_onehot.fit(gbdt.apply(X_train)[:,:,0])
    gbdt_lr = LogisticRegression()
    gbdt_lr.fit(model_onehot.transform(gbdt.apply(X_train)[:,:,0]), y_train)
    return gbdt_lr,model_onehot,gbdt

def GBDTLR_Predict(gbdt_lr,model_onehot,gbdt,X_test):
    pred = gbdt_lr.predict_proba(model_onehot.transform(gbdt.apply(X_test)[:,:,0]))[:, 1]
    return pred


def evaluate_cv(X_train,y_train,model,pars,fold_num = 5,to_balance = False,num_round = 100):
    try:
        X_train = X_train.values
        y_train = y_train.values
    except:
        pass
    #from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=310)
    ks_value_list = []
    auc_value_list = []
    for i,(train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        
        train_x = X_train[train_index]
        train_y = y_train[train_index]
        if to_balance == True:
            train_x,train_y = balance_data(train_x,train_y)
        test_x = X_train[test_index]
        test_y = y_train[test_index]
        if model == 'gbdt':
            gbdt = GBDT_Fit(train_x,train_y,pars)
            test_y_predict = GBDT_Predict(gbdt,test_x)
        elif model == 'xgb':
            xgb_model = XGBoost_Fit(train_x,train_y,pars,num_round = num_round ,X_val=test_x,y_val=test_y)
            test_y_predict = XGBoost_Predict(xgb_model,test_x)
        elif model == 'lr':
            lr_model,scaler = LR_Fit(train_x,train_y,pars)
            test_y_predict = LR_Predict(lr_model,scaler,test_x)
        elif model == 'gbdt_lr' :
            gbdt_lr,model_onehot,gbdt = GBDTLR_Fit(train_x,train_y,pars)
            test_y_predict = GBDTLR_Predict(gbdt_lr,model_onehot,gbdt,test_x)   
        elif model == 'rf':
            rf = RF_Fit(train_x,train_y,pars)
            test_y_predict = RF_Predict(rf,test_x)

        ks_value = ks(test_y,test_y_predict)['ks']
        auc_value = roc_auc_score(test_y, test_y_predict)
        ks_value_list.append(ks_value)
        auc_value_list.append(auc_value)
        print 'now fold %d , all %d flods , ks : %.3f , auc : %.3f'%(i+1,fold_num,ks_value,auc_value) 
    
    ks_mean = np.mean(ks_value_list)
    ks_std = np.std(ks_value_list)
    auc_mean = np.mean(auc_value_list )
    auc_std = np.std(auc_value_list )
    #print 'cv | ks mean : %.3f , ks std : %.3f , auc mean : %.4f'%(ks_mean,ks_std,auc_mean)

    cv_result = 'cv | ks mean : %.3f , ks std : %.3f , auc mean : %.4f , auc std : %.4f'%(ks_mean,ks_std,auc_mean,auc_std)
    return cv_result


def grid_search_pars(X_train,y_train,model,param_grid):
    ks_max = 0
    best_pars = None
    for pars in grid_search(param_grid):
        print pars
        ks_mean= evaluate_cv(X_train,y_train,model=model,pars = pars,fold_num = 5,to_balance = False,num_round = 100)
        if ks_mean > ks_max:
            ks_max = ks_mean
            print ks_max
            best_pars = pars
    return best_pars,ks_max

def find_best_pars(X_train,y_train,what='gbdt'):
    if what == 'gbdt':
        
        gbdt_pars_grid = { 'learning_rate':[0.1,0.2],
                'n_estimators':[50,100],
                'max_depth':[3,4],
                'min_samples_split':[50,100],
                'subsample':[0.8,1],
                'random_state':[10]
                }
    
        best_pars,gbdt_ks_max = grid_search_pars(X_train,y_train,model='gbdt',param_grid=gbdt_pars_grid)

    elif what == 'xgb':
    
        xgb_pars_grid = {
                'eta': [0.1,0.2],
                'gamma':[1,0],
                'min_child_weight':[3,4],
                'max_depth':[4,5],
                'subsample':[0.9],
                'colsample_bytree':[0.9],
                'tree_method': ['approx'],
                'objective': ['binary:logistic'],
                'nthread': [8],
                'seed': [10],
                }
    
        best_pars,xgb_ks_max = grid_search_pars(X_train,y_train,model='xgb',param_grid=xgb_pars_grid)

        '''
        xgb_best_pars = {'colsample_bytree': 0.9,
                         'eta': 0.1,
                         'gamma': 0,
                         'max_depth': 5,
                         'min_child_weight': 4,
                         'nthread': 8,
                         'objective': 'binary:logistic',
                         'seed': 42,
                         'subsample': 0.9,
                         'tree_method': 'approx'}
        '''
    return best_pars


def get_X_y(data,feature_use,source,way=''):

    data_feature_value_miss = count_missing_nan(data[feature_use],1)
    data = data[data_feature_value_miss < len(feature_use)]
    
    data.index = data.source
    data_source = data.loc[source,:]
    
    if way == 'train':
        data_source = pd.concat([data_source,data[data.source == 'zzjr'].sample(frac = 0.2)])
        data_source = pd.concat([data_source,data[data.source == 'yqb'].sample(frac = 0.2)])
    
    X = data_source[feature_use]
    y = data_source['label']
    X = X.fillna(-1)
    X = deal_with_error(X)
    return X,y


def count_missing_nan(data,axis = 1):        
    miss_num = data.isnull().sum(axis)
    return miss_num

def drop_some_data(data,threshold,target = 0,want = 'data'):
    drop_feature = []
    data_len = len(data)
    for f in data.columns:
        ratio = data[f].tolist().count(0) * 1.0 / data_len
        if ratio > threshold:
            drop_feature.append(f)
    if want == 'data':
        return data.drop(drop_feature,axis=1)
    else:                
        return drop_feature

#-------------------------------------- Read Data #--------------------------------------

#--------------------Old Feature--------------------------
old_feature = pd.read_csv('../Data/has_header_feature_all_sample_0623_0706',sep='\t',na_values='-1')
f = open('../Data/tj_feature_new_feature_name','r')
tj_feature_new_feature_name = [x + '_filter' for x in f.readline().split('\t')]
tj_feature_with_filter = pd.read_csv('../Data/tj_feature_label_with_filter',sep='\t',names = ['mbl_num','loan_dt','label','source'] +tj_feature_new_feature_name )
del tj_feature_with_filter['label']

qa_feature_name = [x for x in old_feature if x.startswith('qa')]
apply_feature_name = [x for x in old_feature if x.startswith('apply')]
order_feature_name = [x for x in old_feature if x.startswith('order')]
risk_order_feature_name = [x for x in old_feature if x.startswith('aprv')]
tj_feature_name = [x for x in old_feature if x.startswith('tj')]
tj_filter_feature_name = tj_feature_new_feature_name

#-------------------TJY Feature--------------------------

f = open('../Data/tjy_feature_name','r')
tjy_feature_name = f.readline().split('\t')
tjy_feature = pd.read_csv('../Data/tjy_feature',sep='\t',names = ['mbl_num','loan_dt','label','source'] +tjy_feature_name )
del tjy_feature['label']
tjy_feature = tjy_feature[count_missing_nan(tjy_feature,1)<len(tjy_feature_name)]



f = open('../Data/tjy_log_feature_name','r')
tjy_log_feature_name = f.readline().split('\t')
tjy_log_feature = pd.read_csv('../Data/tjy_log_feature',sep='\t',names = ['mbl_num','loan_dt','label','source'] +tjy_log_feature_name )
del tjy_log_feature['label']

#---------------------lizi Feature-----------------------------------------

f = open('../Data/lida_header','r')
lida_feature_name = [x.strip() for x in f.readline().split(',')]
lida_feature = pd.read_csv('../Data/lida_feature','\t',names = ['mbl_num','loan_dt','label','source']+lida_feature_name)
del lida_feature['label']
lida_feature = drop_some_data(lida_feature,0.95,0)
lida_feature_name = [x for x in lida_feature.columns.tolist()  if x not in ('mbl_num','loan_dt','source','name','id_card','label')]

f = open('../Data/lida_risk_header.txt','r')
lida_risk_feature_name = ['lida_risk_' + x.strip() for x in f.readline().split(',')]
lida_risk_feature = pd.read_csv('../Data/lida_risk_feature','\t',names = ['mbl_num','loan_dt','label','source']+lida_risk_feature_name)
del lida_risk_feature['label']
lida_risk_feature = drop_some_data(lida_risk_feature,0.95,0)
lida_risk_feature_name =  [x for x in lida_risk_feature.columns.tolist()  if x not in ('mbl_num','loan_dt','source','name','id_card','label')]

print 'merge data '

feature_all = pd.merge(old_feature,tj_feature_with_filter , on= ['mbl_num','loan_dt','source'] ,how = 'outer')
feature_all = pd.merge(feature_all,tjy_feature , on= ['mbl_num','loan_dt','source'] ,how = 'outer')
feature_all = pd.merge(feature_all,tjy_log_feature , on= ['mbl_num','loan_dt','source'] ,how = 'outer')
feature_all = pd.merge(feature_all,lida_feature , on= ['mbl_num','loan_dt','source'] ,how = 'outer')
feature_all = pd.merge(feature_all,lida_risk_feature , on= ['mbl_num','loan_dt','source'] ,how = 'outer')
feature_all = feature_all.reset_index(drop=1)

print 'feature_all_shape : '
print feature_all.shape 

label_flie = pd.read_csv('../Data/all_sample_0623',sep='\t',names=['name','mbl_num','id_card','label','loan_dt','source'])
data = pd.merge(feature_all,label_flie , on= ['mbl_num','loan_dt','source'] ,how = 'inner')

feature_use = [x for x in feature_all.columns if x not in ('mbl_num','loan_dt','source','name','id_card','label')]

data = data.reset_index(drop=1)

'''
#----------------KS---------------------
feature_ks = []
data.index = data.source
data_for_ks = data.loc[['yizhen','fqgj'],:]
for feature in feature_use:
    ks_dict = ks(data_for_ks['label'],data_for_ks[feature])
    feature_ks.append(ks_dict['ks'])
    plot_ks(ks_dict,feature)
data = data.reset_index(drop=1)
feature_ks_pd = pd.DataFrame({'feature':feature_use,'ks':feature_ks})
feature_ks_pd.to_csv('../Data/Feature_KS.csv',index = False)
#-----------------------------------
'''

def cover_ratio(feature_use,data,label_flie):
    all_label_length = len(label_flie)
    f_d = data[feature_use]
    r = sum(count_missing_nan(f_d,1) < len(feature_use))*1.0/all_label_length
    return r 


#----------------Select Train Data------------------------------

source_list = ['fqgj', 'zzjr', 'yqb', 'rbl', 'ygz', 'yizhen']
train_source = ['yizhen','fqgj']
print 'Train With '
print train_source

#-----------------Select Feature----------------------------------------------

def Select_Feature(feature_name,threshold,train_source,data):
    keep_feature = []
    for feature in feature_name:
        ks_value = ks(data['label'],data[feature])['ks']
        if ks_value >= threshold:
            keep_feature.append(feature)
    return keep_feature

qa_feature_name_select = Select_Feature(qa_feature_name,3,train_source,data)
risk_order_feature_name_select = Select_Feature(risk_order_feature_name,3,train_source,data)
order_feature_name_select = Select_Feature(order_feature_name,3,train_source,data)
apply_feature_name_select = Select_Feature(apply_feature_name,3,train_source,data)
tj_feature_name_select = Select_Feature(tj_feature_name,3,train_source,data)
tj_filter_feature_name_select = Select_Feature(tj_filter_feature_name,3,train_source,data)
tjy_feature_name_select = Select_Feature(tjy_feature_name,3,train_source,data)
tjy_log_feature_name_select = Select_Feature(tjy_log_feature_name,3,train_source,data)
lida_feature_name_select = Select_Feature(lida_feature_name,3,train_source,data)
lida_risk_feature_name_select = Select_Feature(lida_risk_feature_name,3,train_source,data)



feature_use = qa_feature_name_select + risk_order_feature_name_select \
             + order_feature_name_select + apply_feature_name_select \
             + tj_feature_name_select + tj_filter_feature_name_select \
             + tjy_feature_name_select + tjy_log_feature_name_select \
             + lida_feature_name_select + lida_risk_feature_name_select



# qa + risk
feature_use = qa_feature_name_select + risk_order_feature_name_select
# qa + risk + order
feature_use = qa_feature_name_select + risk_order_feature_name_select + order_feature_name_select
# qa + risk + tjy + lizi
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select

# qa + risk + order + tjy + lizi
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select

# qa + risk + order + tjy + lizi +tj

feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select + tj_feature_name_select

# qa + risk + order + tjy + lizi + tj_filter
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select + tj_filter_feature_name_select


# qa + risk + order + tjy + lizi + tj_filter + lizi_risk
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select + tj_filter_feature_name_select \
                + lida_risk_feature_name_select
                
# qa + risk + order + tjy + lizi + tj_filter + lizi_risk +apply
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select + tj_filter_feature_name_select \
                + lida_risk_feature_name_select + apply_feature_name_select

# all
feature_use = qa_feature_name_select + risk_order_feature_name_select + tjy_feature_name_select\
                + tjy_log_feature_name_select + lida_feature_name_select\
                + order_feature_name_select + tj_filter_feature_name_select \
                + lida_risk_feature_name_select + apply_feature_name_select \
                + tj_feature_name_select


print 'Train With %d Features'%(len(feature_use))
print 'Cover Rate : %.3f'%(cover_ratio(feature_use,data,label_flie))
#---------------------------------End Data-----------------

'''
feature_ks = open('../Data/Select_Feature_KS.txt','w')
for feature in feature_use:
    ks_dict = ks(data['label'],data[feature])
    line = print_ks(ks_dict,feature)
    feature_ks.write(line)
    feature_ks.write('---------------------------------------------------------------------\n')
feature_ks.close()
''' 

'''
#---------------------Feature Importance --------------------
X_train,y_train = get_X_y(data,feature_use,train_source)


pars_for_rf = {'n_estimators':50}

rf = RF_Fit(X_train,y_train,pars_for_rf)

feature_score = sorted(zip(X_train.columns,rf.feature_importances_),key=lambda x:x[1],reverse=True)
feature_score_pd = pd.DataFrame(feature_score,columns = ['feature_name','rf_score'])
feature_score_pd.to_csv('../Data/feature_importance.csv',index = False)
feature_select_num = input('preserve how many feature ? input a number : ')
feature_use = [x[0] for x in feature_score[:feature_select_num]]

'''

#-----------------Train Model-----------------------

X_train,y_train = get_X_y(data,feature_use,train_source)#,way='train')

'''
lr_best_pars = {'penalty': 'l1', 'C': 0.1}
lr_model,scaler = LR_Fit(X_train,y_train,lr_best_pars)
feature_use = np.array(feature_use)[lr_model.coef_[0] >0].tolist()
X_train,y_train = get_X_y(data,feature_use,train_source)
'''



# XGB
xgb_best_pars = {'colsample_bytree': 0.9,
                 'eta': 0.1,
                 'gamma': 0,
                 'max_depth': 5,
                 'min_child_weight': 4,
                 'nthread': 8,
                 'objective': 'binary:logistic',
                 'seed': 42,
                 'subsample': 0.9,
                 'tree_method': 'approx'}

cv_result = evaluate_cv(X_train,y_train,model='xgb',pars=xgb_best_pars,fold_num = 5,to_balance = False,num_round = 60)
print cv_result
xgb_model = XGBoost_Fit(X_train,y_train,xgb_best_pars,num_round=60)
ks_all = {}
for test_source in source_list:
    print test_source
    X_test,y_test = get_X_y(data,feature_use,test_source)
    pred = XGBoost_Predict(xgb_model,X_test)
    ks_result = ks(y_test,pred)
    ks_all[test_source] = ks_result

for source,ks_value in ks_all.items():
    print source
    print ks_value['ks']

gc.collect()


# GBDT
print '-------------------------GBDT Result-------------------------'
gbdt_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8,'random_state':10}
cv_result = evaluate_cv(X_train,y_train,model='gbdt',pars=gbdt_best_pars,fold_num = 5,to_balance = False,num_round = 60)
print cv_result
gbdt_model = GBDT_Fit(X_train,y_train,gbdt_best_pars)
ks_all = {}
pred_list = []
for test_source in source_list:
    X_test,y_test = get_X_y(data,feature_use,test_source)
    pred = GBDT_Predict(gbdt_model,X_test)
    pred_list.append(pred)
    ks_result = ks(y_test,pred)
    ks_all[test_source] = ks_result
for source,ks_value in ks_all.items():
    #print_ks(ks_value,source)
    print source
    print ks_value['ks']
gc.collect()


#import seaborn as sns
#import matplotlib.pyplot as plt
#for i,s in enumerate(source_list):
#    sns.kdeplot(pred_list[i],label=s)
#plt.legend()


'''
# GBD_LR
gbdt_lr_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8,'random_state':10}
cv_result = evaluate_cv(X_train,y_train,model='gbdt_lr',pars = gbdt_lr_best_pars, fold_num = 5)
print cv_result
gbdt_lr,model_onehot,gbdt = GBDTLR_Fit(X_train,y_train,gbdt_lr_best_pars)
ks_all = {}
for test_source in source_list:
    print test_source
    X_test,y_test = get_X_y(data,feature_use,test_source)
    pred = GBDTLR_Predict(gbdt_lr,model_onehot,gbdt,X_test)
    ks_result = ks(y_test,pred)
    ks_all[test_source] = ks_result
for source,ks_value in ks_all.items():
    print source
    print ks_value['ks']
gc.collect()
'''

'''
# LR
print '-------------------------LR Result-------------------------'
lr_best_pars = {'penalty': 'l1', 'C': 0.1}
cv_result = evaluate_cv(X_train,y_train,model='lr',pars = lr_best_pars, fold_num = 5)
print cv_result
lr_model,scaler = LR_Fit(X_train,y_train,lr_best_pars)
ks_all = {}
for test_source in source_list:
    print test_source
    X_test,y_test = get_X_y(data,feature_use,test_source)
    pred = LR_Predict(lr_model,scaler,X_test)
    ks_result = ks(y_test,pred)
    ks_all[test_source] = ks_result
for source,ks_value in ks_all.items():
    print source
    print ks_value['ks']
gc.collect()
'''






'''
gbdt_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8,'random_state':310}

xgb_best_pars = {'colsample_bytree': 0.9,
                 'eta': 0.1,
                 'gamma': 0,
                 'max_depth': 5,
                 'min_child_weight': 4,
                 'nthread': 8,
                 'objective': 'binary:logistic',
                 'seed': 42,
                 'subsample': 0.9,
                 'tree_method': 'approx'}

lgb_best_pars = {'bagging_fraction': 0.8,
                 'boosting_type': 'gbdt',
                 'colsample_bytree': 0.7,
                 'feature_fraction': 0.9,
                 'learning_rate': 0.1,
                 'max_bin': 255,
                 'metric': 'binary_logloss',
                 'min_child_weight': 1.5,
                 'num_leaves': 10,
                 'objective': 'binary',
                 'seed': 2017,
                 'subsample': 0.7,
                 'verbose': 0}
gbdt_lr_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8,'random_state':310}
'''
