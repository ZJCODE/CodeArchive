#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zhangjun
"""
from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter
from get_ks import ks,print_ks,plot_ks,print_ks_
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import xgboost as xgb


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
    pred = rf.predict(X_test)
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


def GBDTLR_Fit(X_train,y_train,pars):
    #from sklearn.ensemble import GradientBoostingClassifier
    #from sklearn.preprocessing import OneHotEncoder
    gbdt = GradientBoostingClassifier(**pars)
    gbdt.fit(X_train,y_train)
    model_onehot = OneHotEncoder()
    model_onehot.fit(gbdt.apply(X_train)[:,:,0])
    gbdt_lr = LogisticRegression(penalty='l2')
    gbdt_lr.fit(model_onehot.transform(gbdt.apply(X_train)[:,:,0]), y_train)
    return gbdt_lr,model_onehot,gbdt

def GBDTLR_Predict(gbdt_lr,model_onehot,gbdt,X_test):
    pred = gbdt_lr.predict_proba(model_onehot.transform(gbdt.apply(X_test)[:,:,0]))[:, 1]
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




def evaluate_cv(X_train,y_train,model,pars,fold_num = 5,num_round = 100,show_print = False):
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
        if show_print:
            print 'now fold %d , all %d flods , ks : %.3f , auc : %.3f'%(i+1,fold_num,ks_value,auc_value) 
    
    ks_mean = np.mean(ks_value_list)
    ks_std = np.std(ks_value_list)
    auc_mean = np.mean(auc_value_list )
    auc_std = np.std(auc_value_list )
    #print 'cv | ks mean : %.3f , ks std : %.3f , auc mean : %.4f'%(ks_mean,ks_std,auc_mean)

    cv_result = 'cv | ks mean : %.3f , ks std : %.3f , auc mean : %.4f , auc std : %.4f'%(ks_mean,ks_std,auc_mean,auc_std)
    return cv_result


