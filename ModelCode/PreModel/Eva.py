# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:15:55 2017

@author: zhangjun
"""

import pandas as pd
from get_ks import ks,print_ks,plot_ks

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

f
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


#  特征重要性

def feature_summary(data,feature_use,split_source = False):
    feature_use_nan = pd.DataFrame(feature_use)
    feature_zero = pd.DataFrame(feature_use)
    feature_ks = pd.DataFrame(feature_use)
    
    data_length = len(data)
    feature_use_nan_list = []
    feature_zero_list = []
    feature_ks_list = []
    for f in feature_use:
        feature_use_nan_list.append(sum(data[f].isnull())*1.0/data_length)
        feature_zero_list.append(sum(data[f] == 0)*1.0/data_length)
        ks_dict = ks(data['label'],data[f])
        feature_ks_list.append(ks_dict['ks'])
        plot_ks(ks_dict,f,'all')
    feature_use_nan['all'] = feature_use_nan_list
    feature_zero['all'] = feature_zero_list
    feature_ks['all'] = feature_ks_list
    
    
    if split_source:
        source = data.source.unique()
        for s in source:
            data_s = data[data.source == s]
            data_s_length = len(data_s)
            feature_use_nan_list = []
            feature_zero_list = []
            feature_ks_list = []
            for f in feature_use:
                feature_use_nan_list.append(sum(data_s[f].isnull())*1.0/data_s_length)
                feature_zero_list.append(sum(data_s[f] == 0)*1.0/data_s_length)
                ks_dict = ks(data_s['label'],data_s[f])
                feature_ks_list.append(ks_dict['ks'])
                plot_ks(ks_dict,f,s)
            feature_use_nan[s] = feature_use_nan_list
            feature_zero[s] = feature_zero_list
            feature_ks[s] = feature_ks_list
        
    return feature_use_nan,feature_zero,feature_ks

feature_use_nan,feature_zero,feature_ks = feature_summary(data,feature_use,split_source=True)

feature_use_nan.to_csv('../Data/feature_use_nan.csv')
feature_zero.to_csv('../Data/feature_zero.csv') 
feature_ks.to_csv('../Data/feature_ks.csv')


feature_ks = open('../Data/Feature_KS_Detail.txt','w')
for feature in feature_use:
    ks_dict = ks(data['label'],data[feature])
    line = print_ks(ks_dict,feature)
    feature_ks.write(line)
    feature_ks.write('---------------------------------------------------------------------\n')
feature_ks.close()

'''
import seaborn as sns
import matplotlib.pyplot as plt
for i,s in enumerate(source_list):
    sns.kdeplot(pred_list[i],label=s)
plt.legend()
'''