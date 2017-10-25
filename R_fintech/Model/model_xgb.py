#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zhangjun
"""
import pandas as pd
import numpy as np
from collections import Counter
import time
from Model_Files import *
from get_ks import ks,print_ks
import gc
import sys
from sklearn.metrics import roc_auc_score

log_text = ''

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]

log_path = 'log/result_xgb_'+str(int(time.time()))+'.log'
pred_path = 'pred/pred_' + str(int(time.time())) 
log_text = log_text + 'train :' + train_data_path + '\n'
log_text = log_text + 'test : ' + test_data_path + ' | base_test_feature ' + '\n'


def load_data(path,resample=False):
    # data 前面特征，后三列  ['label','source','loan_dt']
    data =pd.read_csv(path,'\t')
    data = data.fillna(-1)
    return data

def data_resample(data,good_times = 10):
    default_bad_good_ratio = 1.0/good_times
    sources = data.source.unique()
    data_resample = data.head()
    for s in sources:
        data_temp = data[data.source == s]
        overdue_data = data_temp[data_temp.label==1]
        good_data = data_temp[data_temp.label==0]
        num_overdue = len(overdue_data)
        num_good = int(num_overdue/default_bad_good_ratio)
        good_data_resample = good_data.sample(num_good,replace=True)
        data_resample = pd.concat([data_resample,overdue_data])
        data_resample = pd.concat([data_resample,good_data_resample])
    return data_resample

def split_based_on_source_time(data,test_ratio=0.3,without_old = True):
    # 每个样本都按时间留出一部分当测试集
    if without_old:
        data = data[data.loan_dt >= '2017-02-01']
    source_list = data.source.unique()
    data_train = data.head(0)
    data_test = data.head(0)
    for s in source_list:
        #print 'process %s'%(s)
        data_s = data[data.source == s]
        data_s = data_s.sort_values('loan_dt')
        data_train = pd.concat([data_train,data_s[:int(len(data_s)*(1-test_ratio))]])
        data_test = pd.concat([data_test,data_s[int(len(data_s)*(1-test_ratio)):]])
    data_train = data_train.reset_index(drop=1)
    data_test = data_test.reset_index(drop=1)
    return data_train,data_test


def select_feature(data,top_num = 150,without_old=True):
    # 按时效性筛选特征 
    # data 前面特征，后三列  ['label','source','loan_dt']
    if without_old:
        data = data[data.loan_dt >= '2017-02-01']
    time_range = [x[:10] for x in map(str,pd.date_range(data.loan_dt.min(),data.loan_dt.max(),freq='15D'))]
    print time_range
    pars_for_gbdt = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 20,'subsample': 0.8,'random_state':10}
    feature_importance_pd = pd.DataFrame({'feature':data.columns[:-3]})
    for num in range(len(time_range)-1):
        #print 'round : %d'%(num)
        data_t = data[(data.loan_dt >= time_range[num])&(data.loan_dt <= time_range[num+1])]
        X_train = data_t.iloc[:,:-3]
        y_train = data_t['label'].values
        gbdt_model = GBDT_Fit(X_train,y_train,pars_for_gbdt)
        feature_importance_gbdt = pd.DataFrame({'feature':X_train.columns,'feature_importance_gbdt_'+ time_range[num] :gbdt_model.feature_importances_})
        feature_importance_pd = pd.merge(feature_importance_pd,feature_importance_gbdt,on=['feature'])
    feature_importance_pd.to_csv('sample_time_feature_importance.csv',index=False)
    #time_weight = np.ones(len(time_range)-1).reshape(-1,1)
    time_weight = np.arange(len(time_range)-1).reshape(-1,1)
    feature_importance_pd['combine'] = np.dot(feature_importance_pd.iloc[:,1:].values,time_weight)
    feature_importance_pd = feature_importance_pd.sort_values('combine',ascending=False)
    feature_select = feature_importance_pd.feature[:top_num].values.tolist() + ['label','source','loan_dt']
    return feature_select

def data_describe(data):
    data_desc = pd.DataFrame({'source':data.pivot_table('loan_dt','source',aggfunc='max').index,\
                 'sample_cnt':data.pivot_table('label','source',aggfunc='count').values.reshape(1,-1)[0] ,\
                'overdue_rate':data.pivot_table('label','source',aggfunc='mean').values.reshape(1,-1)[0] ,\
                  't_min':pd.to_datetime(data.pivot_table('loan_dt','source',aggfunc='min').values.reshape(1,-1)[0]) ,\
                  't_max':pd.to_datetime(data.pivot_table('loan_dt','source',aggfunc='max').values.reshape(1,-1)[0])})
    data_desc = data_desc[['source','sample_cnt','overdue_rate','t_min','t_max']]
    data_desc['t_range'] = data_desc.t_max - data_desc.t_min
    print data_desc
    return data_desc

#----------------------------------Prepare Data------------------------------------------------------

print 'train with : ' + train_data_path
print 'test with  : ' + test_data_path + ' | base_test_feature '

data_train = load_data(train_data_path)
data_test_raw = load_data(test_data_path)
ygz_test = load_data('ygz/ygz_30_feature')
data_test_raw = pd.concat([data_test_raw,ygz_test])
base_test = load_data('outer/base_test_feature')
data_test = pd.concat([data_test_raw,base_test])
#data_test  = data_test[(data_test.loan_dt > '2017-01-01')&(data_test.loan_dt < '2017-07-20')]

cross_time_test = sys.argv[3] #input('cross time test ? [0/1] : ')
cross_time_test = int(cross_time_test)
if cross_time_test == 1:
	data_test_copy = data_test.copy()
	data_test_copy.source = [s+'_'+x[:7] for s,x in zip(data_test_copy.source,data_test_copy.loan_dt)]
    data_test = pd.concat([data_test,data_test_copy])


test_ratio = sys.argv[4] #input('test_ratio : ')
test_ratio = float(test_ratio)
if test_ratio < 1:
	data_train_add,data_test = split_based_on_source_time(data_test,test_ratio = test_ratio ,without_old = True)
    data_train = pd.concat([data_train,data_train_add])
log_text = log_text + 'test ratio %f'%(test_ratio) + '\n'

data_test = pd.concat([data_test,base_test])

#data_test  = data_test[(data_test.loan_dt > '2017-01-01')&(data_test.loan_dt < '2017-07-20')]

print '----------------------------------'
print 'train data describe : '
print '----------------------------------'
train_data_describe = data_describe(data_train)
print '----------------------------------'
print 'test data describe : '
print '----------------------------------'
test_data_describe = data_describe(data_test)

log_text = log_text + '-----------\ntrain data describe\n' + str(train_data_describe) + '\n'
log_text = log_text + '-----------\ntest data describe\n' + str(test_data_describe) + '\n'

good_times = sys.argv[5] 
good_times = float(good_times)
if good_times > 0 :
    print 'resample data'
    log_text = log_text + 'resample data \n'
    data_train = data_resample(data_train,good_times = good_times)
    print '----------------------------------'
    print 'resample train data describe : '
    print '----------------------------------'
    train_data_describe = data_describe(data_train)
    log_text = log_text + '-----------\nresample train data describe\n' + str(train_data_describe) + '\n'


top_num = sys.argv[6] #input('select feature num :')
top_num = int(top_num)
if top_num > 0:
    feature_select = select_feature(data_train,top_num = top_num)
else:
    feature_select  = data_train.columns

print 'use %d features'%(len(feature_select)-3)

log_text = log_text + 'use %d features'%(len(feature_select)-3) + '\n'

data_train_select = data_train[feature_select] 
data_test_select = data_test[feature_select]

def get_X_y(data):
    # data 前面特征，后三列  ['label','source','loan_dt']
    X = data.iloc[:,:-3].values
    y = data['label'].values
    return X,y

#------------------------------------------- ModelPart-------------------------------------

X_train,y_train = get_X_y(data_train_select)

xgb_pars_default = sys.argv[7] # input('gbdt_pars_default : [0/1] ')
xgb_pars_default = int(xgb_pars_default)
if xgb_pars_default == 0:
	pass
else:
    xgb_pars = {'colsample_bytree': 0.9,
                 'eta': 0.2,
                 'gamma': 0,
                 'max_depth': 4,
                 'min_child_weight': 4,
                 'nthread': 10,
                 'objective': 'binary:logistic',
                 'seed': 42,
                 'subsample': 0.9,
                 'tree_method': 'approx',
                 'eval_metric':'auc'}

log_text = log_text + str(xgb_pars) + '\n'
log_text = log_text + '----------------------------------\n'
#pars_for_lr = {'penalty': 'l1', 'C': 0.1}
print '----------------------------------'
#print 'tarin gbdt cv result '
print xgb_pars
#cv_result = evaluate_cv(X_train,y_train,model='gbdt',pars=pars_for_gbdt,fold_num = 5)
print '----------------------------------'
#print cv_result
#print '----------------------------------'
#print 'tarin lr cv result '
#cv_result = evaluate_cv(X_train,y_train,model='lr',pars=pars_for_lr,fold_num = 5)
#print '----------------------------------'
#print cv_result

print 'xgb test eval'

test_data_describe_select = test_data_describe[
                   (test_data_describe.sample_cnt > 500)&                   
                   (test_data_describe.overdue_rate != 0)
                   ]

xgb_model = XGBoost_Fit(X_train,y_train,xgb_pars,100)

pred_list = [[]]

test_source = test_data_describe_select.source
for s in test_source:
    pred_list[0].append(s)
    print '----- source : %s   cnt : %d   overdue_rate : %.3f   start_day : %s   end_day : %s -----'%(s,
                                             test_data_describe_select[test_data_describe_select.source == s].sample_cnt.values[0],
                                             test_data_describe_select[test_data_describe_select.source == s].overdue_rate.values[0],
                                             str(test_data_describe_select[test_data_describe_select.source == s].t_min.values[0])[:10],
                                             str(test_data_describe_select[test_data_describe_select.source == s].t_max.values[0])[:10]
                                             )
    log_text = log_text + '----- source : %s   cnt : %d   overdue_rate : %.3f   start_day : %s   end_day : %s -----'%(s,
                                             test_data_describe_select[test_data_describe_select.source == s].sample_cnt.values[0],
                                             test_data_describe_select[test_data_describe_select.source == s].overdue_rate.values[0],
                                             str(test_data_describe_select[test_data_describe_select.source == s].t_min.values[0])[:10],
                                             str(test_data_describe_select[test_data_describe_select.source == s].t_max.values[0])[:10]
                                             ) + '\n'


    data_temp = data_test_select[(data_test_select.source == s)]
    X_test,y_test = get_X_y(data_temp)
    #self_cv = evaluate_cv(X_test,y_test,model='xgb',pars=xgb_pars,fold_num = 5,num_round=60)
    #print 'self cv :'
    #log_text = log_text + 'self cv : \n'
    #print self_cv
    #log_text = log_text + str(self_cv) + '\n'
    pred_xgb = XGBoost_Predict(xgb_model,X_test)
    pred_list.append(pred_xgb)
    ks_result = ks(y_test,pred_xgb)
    auc_value = roc_auc_score(y_test, pred_xgb)
    print 'cross sample :'
    print 'ks : %.3f   auc : %.3f '%(ks_result['ks'],auc_value)
    log_text = log_text + 'cross sample : \n'
    log_text = log_text + 'ks : %.3f   auc : %.3f '%(ks_result['ks'],auc_value) + '\n'
    print print_ks(ks_result)
np.save(pred_path,pred_list)
f = open(log_path,'w')
f.write(log_text)
f.close()
