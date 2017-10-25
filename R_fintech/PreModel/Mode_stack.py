#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:18:08 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
from collections import Counter
from get_ks import ks,print_ks
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
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


# GBDT

def GBDT_Fit(X_train,y_train,pars):
    #from sklearn.ensemble import GradientBoostingClassifier
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

# LightGBM

def LightGBM_Fit(X_train,y_train,pars,num_round,X_val=None,y_val=None):
    #import lightgbm as lgb
    dtrain = lgb.Dataset(X_train, label=y_train)
    try:
        _ = len(X_val)
        dval = lgb.Dataset(X_val, label=y_val)
        lgb_model= lgb.train(pars, dtrain, verbose_eval=1,valid_sets=dval, num_boost_round = num_round,early_stopping_rounds = 15)
    except:
        lgb_model= lgb.train(pars, dtrain, verbose_eval=1,num_boost_round = num_round)
    return lgb_model

def LightGBM_Predict(lgb_model,X_test):
    pred = lgb_model.predict(X_test,data_has_header=False)
    return pred

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

# StackModel

def StackModel_Fit(X_train,y_train,gbdt_pars,xgb_pars,lgb_pars,stacking_model,stack_fold = 2):
    try:
        X_train = X_train.values
        y_train = y_train.values
    except:
        pass
    skf = StratifiedKFold(n_splits=stack_fold, shuffle=True, random_state=310)
    base_model_num = 3
    layer_train = np.zeros((X_train.shape[0], base_model_num))

    print 'begin gbdt model'
    gbdt_model_list = []
    for k,(train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        train_x = X_train[train_index]
        train_y = y_train[train_index]
        test_x = X_train[test_index]
        gbdt_model = GBDT_Fit(train_x,train_y,gbdt_pars)
        pre_y = GBDT_Predict(gbdt_model,test_x)
        layer_train[test_index, 0] = pre_y
        gbdt_model_list.append(gbdt_model)
    
    print 'begin xgboost model'
    xgb_model_list = []
    for k,(train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        train_x = X_train[train_index]
        train_y = y_train[train_index]
        test_x = X_train[test_index]
        xgb_model = XGBoost_Fit(train_x,train_y,xgb_pars,num_round=60)
        pre_y = XGBoost_Predict(xgb_model,test_x)
        layer_train[test_index, 1] = pre_y
        xgb_model_list.append(xgb_model)


    print 'begin lgb model'
    lgb_model_list = []
    for k,(train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        train_x = X_train[train_index]
        train_y = y_train[train_index]
        test_x = X_train[test_index]
        lgb_model = LightGBM_Fit(train_x,train_y,lgb_pars,num_round=60)
        pre_y = LightGBM_Predict(lgb_model,test_x)
        layer_train[test_index, 2] = pre_y
        lgb_model_list.append(lgb_model)

    stacking_model.fit(layer_train,y_train)

    return gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model

def StackModel_Predict(gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model,X_test):
    try:
        X_test = X_test.values
    except:
        pass
    base_model_num = 3
    fold_num = len(gbdt_model_list)
    layer_test = np.zeros((X_test.shape[0], base_model_num))

    print 'begin gbdt model'
    k_model_i = np.zeros((X_test.shape[0], fold_num))
    for k,gbdt_model in enumerate(gbdt_model_list):
        k_model_i[:, k] = GBDT_Predict(gbdt_model,X_test)
    layer_test[:, 0] = k_model_i.mean(axis=1)

    print 'begin xgboost model'
    k_model_i = np.zeros((X_test.shape[0], fold_num))
    for k,xgb_model in enumerate(xgb_model_list):
        k_model_i[:, k] = XGBoost_Predict(xgb_model,X_test)
    layer_test[:, 1] = k_model_i.mean(axis=1)

    print 'begin lgb model'
    k_model_i = np.zeros((X_test.shape[0], fold_num))
    for k,lgb_model in enumerate(lgb_model_list):
        k_model_i[:, k] = LightGBM_Predict(lgb_model,X_test)
    layer_test[:, 2] = k_model_i.mean(axis=1)

    pred = stacking_model.predict_proba(layer_test)[:,1]

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
        elif model == 'lgb' :
            lgb_model = LightGBM_Fit(train_x,train_y,pars,num_round = num_round ,X_val=test_x,y_val=test_y)
            test_y_predict = LightGBM_Predict(lgb_model,test_x)
        elif model == 'gbdt_lr' :
            gbdt_lr,model_onehot,gbdt = GBDTLR_Fit(train_x,train_y,pars)
            test_y_predict = GBDTLR_Predict(gbdt_lr,model_onehot,gbdt,test_x)

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

def evaluate_stack_cv(X_train,y_train,gbdt_pars,xgb_pars,lgb_pars,stacking_model,to_balance = False,fold_num = 5,stack_fold = 2):
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

        gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model = StackModel_Fit(train_x,train_y,
            gbdt_pars,xgb_pars,lgb_pars,stacking_model,stack_fold = stack_fold)

        test_y_predict = StackModel_Predict(gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model,test_x)

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

def find_best_pars(X_train,y_train):

    gbdt_pars_grid = { 'learning_rate':[0.1,0.2],
            'n_estimators':[50,100],
            'max_depth':[3,4],
            'subsample':[0.8,1],
            'random_state':[310]
            }

    gbdt_best_pars,gbdt_ks_max = grid_search_pars(X_train,y_train,model='gbdt',param_grid=gbdt_pars_grid)

    '''
    gbdt_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8,'random_state':310}
    '''


    xgb_pars_grid = {
            'eta': [0.1,0.2],
            'gamma':[1,0],
            'min_child_weight':[3,4],
            'max_depth':[5,6],
            'subsample':[0.9],
            'colsample_bytree':[0.9],
            'tree_method': ['approx'],
            'objective': ['binary:logistic'],
            'nthread': [8],
            'seed': [2017],
            }

    xgb_best_pars,xgb_ks_max = grid_search_pars(X_train,y_train,model='xgb',param_grid=xgb_pars_grid)

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


    lgb_pars_grid = {

            'boosting_type': ['gbdt'],
            'metric': ['binary_logloss'],
            'objective': ['binary'],
            #'objective': 'regression',
            'min_child_weight': [1.5,1],
            'num_leaves': [12,10],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'seed': [2017],
            'learning_rate': [0.1,0.2],
            'feature_fraction': [0.9],
            'bagging_fraction': [0.8],
            'verbose': [0]
            }

    lgb_best_pars,xgb_ks_max = grid_search_pars(X_train,y_train,model='lgb',param_grid=lgb_pars_grid)

    '''

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

     '''



    gbdt_lr_pars_grid = { 'learning_rate':[0.1,0.2],
            'n_estimators':[50,100],
            'max_depth':[3,4],
            'subsample':[0.8,1],
            'random_state':[310]
            }


    gbdt_lr_best_pars,gbdt_lr_ks_max = grid_search_pars(X_train,y_train,model='gbdt_lr',param_grid=gbdt_lr_pars_grid)

    '''
    gbdt_lr_best_pars = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8,'random_state':310}
    '''

    return gbdt_best_pars,xgb_best_pars,lgb_best_pars,gbdt_lr_best_pars

def get_X_y(data,feature_use,source,fillna_way):
    data_source = data[data.source == source]
    X = data_source[feature_use]
    y = data_source['label']
    X = deal_with_error(X)
    X,_ = fill_nan(X,y,way=fillna_way)
    return X,y



# Read Data
feature = pd.read_csv('../Data/has_header_feature_all_sample_0623_0706',sep='\t',na_values='-1')
label_flie = pd.read_csv('../Data/all_sample_0623',sep='\t',names=['name','mbl_num','id_card','label','loan_dt','source'])
data = pd.merge(feature,label_flie , on= ['mbl_num','loan_dt','source'] ,how = 'left')
data = data.sort_values('loan_dt')
feature_use = [u'qa_op_free', u'qa_op_student',
       u'qa_col_type_xcq', u'qa_col_type_fgf', u'qa_col_type_zjf',
       u'qa_has_car', u'qa_no_social_security', u'qa_user_income',
       u'apply_cnt', u'apply_limit_mode', u'apply_cnt_std_3715306090180',
       u'apply_night_pct', u'apply_cnt_90d', u'apply_cnt_31-90d', u'order_cnt',
       u'order_mode', u'order_bank_cnt', u'order_status_cnt',
       u'order_cnt_std_3715306090180', u'order_limit_pct_std_1k1w2w',
       u'order_night_pct', u'order_avg', u'order_max', u'order_min',
       u'order_cnt_15d', u'order_cnt_90d', u'aprv_cnt', u'aprv_pct_15d',
       u'aprv_pct_60d', u'aprv_pct_90d', u'aprv_apply_limit_pct_1k',
       u'aprv_apply_limit_pct_2k', u'aprv_limit_pct_0.5k',
       u'aprv_limit_pct_1k', u'aprv_reject_cnt', u'aprv_pre_repay',
       u'aprv_repay', u'aprv_limit_max', u'aprv_gap_std',
       u'aprv_limit_positive_sum', u'aprv_limit_mean', u'aprv_limit_var',
       u'aprv_unnomal_cnt', u'aprv_overdue_last_gap', u'tj_cnt',
       u'tj_user_cnt', u'tj_resource_cnt', u'tj_cnt_std_3715306090',
       u'tj_fee_cnt', u'tj_cnt_90d']

#----------------------------------------

# 'fqgj', 'zzjr', 'yqb', 'rbl', 'ygz', 'yizhen'

source_list = ['fqgj', 'zzjr', 'yqb', 'rbl', 'ygz', 'yizhen']

train_source = 'fqgj'

#----------------------------------------


# LGB

X_train,y_train = get_X_y(data,feature_use,train_source,fillna_way=-1)
cv_result = evaluate_cv(X_train,y_train,model='lgb',pars=lgb_best_pars,fold_num = 5,to_balance = True,num_round = 60)
lgb_model = LightGBM_Fit(X_train,y_train,lgb_best_pars,num_round=60)
ks_all = {}
for test_source in source_list:
    print test_source
    if test_source == train_source:
        pass
    else:
        X_test,y_test = get_X_y(data,feature_use,test_source,fillna_way=-1)
        pred = LightGBM_Predict(lgb_model,X_test)
        ks_result = ks(y_test,pred)
        ks_all[test_source] = ks_result

for source,ks_value in ks_all.items():
    print_ks(ks_value,source)

# XGB

X_train,y_train = get_X_y(data,feature_use,train_source,fillna_way=-1)
cv_result = evaluate_cv(X_train,y_train,model='xgb',pars=xgb_best_pars,fold_num = 5,to_balance = True,num_round = 60)
xgb_model = XGBoost_Fit(X_train,y_train,xgb_best_pars,num_round=60)
ks_all = {}
for test_source in source_list:
    print test_source
    if test_source == train_source:
        pass
    else:
        X_test,y_test = get_X_y(data,feature_use,test_source,fillna_way=-1)
        pred = XGBoost_Predict(xgb_model,X_test)
        ks_result = ks(y_test,pred)
        ks_all[test_source] = ks_result

for source,ks_value in ks_all.items():
    print_ks(ks_value,source)



# GBDT

X_train,y_train = get_X_y(data,feature_use,train_source,fillna_way=-1)
cv_result = evaluate_cv(X_train,y_train,model='gbdt',pars=gbdt_best_pars,fold_num = 5,to_balance = True,num_round = 60)
gbdt_model = GBDT_Fit(X_train,y_train,gbdt_best_pars)
ks_all = {}
for test_source in source_list:
    print test_source
    if test_source == train_source:
        pass
    else:
        X_test,y_test = get_X_y(data,feature_use,test_source,fillna_way=-1)
        pred = GBDT_Predict(gbdt_model,X_test)
        ks_result = ks(y_test,pred)
        ks_all[test_source] = ks_result

for source,ks_value in ks_all.items():
    print_ks(ks_value,source)


# STACK
stacking_model = LogisticRegression()
X_train,y_train = get_X_y(data,feature_use,train_source,fillna_way=-1)
cv_result = evaluate_stack_cv(X_train,y_train,gbdt_pars=gbdt_best_pars,xgb_pars=xgb_best_pars,lgb_pars=lgb_best_pars,stacking_model=stacking_model,to_balance = False,fold_num = 5,stack_fold=3)
gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model =StackModel_Fit(X_train,y_train,gbdt_pars=gbdt_best_pars,xgb_pars=xgb_best_pars,lgb_pars=lgb_best_pars,stacking_model=stacking_model,stack_fold = 3)
ks_all = {}
for test_source in source_list:
    print test_source
    if test_source == train_source:
        pass
    else:
        X_test,y_test = get_X_y(data,feature_use,test_source,fillna_way=-1)
        pred = StackModel_Predict(gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model,X_test)
        ks_result = ks(y_test,pred)
        ks_all[test_source] = ks_result

for source,ks_value in ks_all.items():
    print_ks(ks_value,source)


gc.collect()


#gbdt_best_pars,xgb_best_pars,lgb_best_pars,gbdt_lr_best_pars = find_best_pars(X_train,y_train)


cv_result = evaluate_cv(X_train,y_train,model='lgb',pars=lgb_best_pars,fold_num = 5,to_balance = True,num_round = 60)
lgb_model = LightGBM_Fit(X_train,y_train,lgb_best_pars,num_round=60)
pred = LightGBM_Predict(lgb_model,X_test)
ks(y_test,pred)


cv_result = evaluate_cv(X_train,y_train,model='xgb',pars=xgb_best_pars,fold_num = 5,to_balance = True,num_round = 60)
xgb_model = XGBoost_Fit(X_train,y_train,xgb_best_pars,num_round=60)
pred = XGBoost_Predict(xgb_model,X_test)
ks(y_test,pred)

cv_result = evaluate_cv(X_train,y_train,model='gbdt',pars=gbdt_best_pars,fold_num = 5,to_balance = True,num_round = 60)
gbdt_model = GBDT_Fit(X_train,y_train,gbdt_best_pars)
pred = GBDT_Predict(gbdt_model,X_test)
ks(y_test,pred)

#----------------------------------------

stacking_model = LogisticRegression()
cv_result = evaluate_stack_cv(X_train,y_train,gbdt_pars=gbdt_best_pars,xgb_pars=xgb_best_pars,lgb_pars=lgb_best_pars,stacking_model=stacking_model,to_balance = False,fold_num = 5,stack_fold=2)
gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model =StackModel_Fit(X_train,y_train,gbdt_pars=gbdt_best_pars,xgb_pars=xgb_best_pars,lgb_pars=lgb_best_pars,stacking_model=stacking_model,stack_fold = 2)
pred = StackModel_Predict(gbdt_model_list,xgb_model_list,lgb_model_list,stacking_model,X_test)
ks(y_test,pred)








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
