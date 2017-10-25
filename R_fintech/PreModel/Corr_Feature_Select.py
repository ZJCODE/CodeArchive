# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:51:06 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
from get_ks import ks 

def cos_sim(X):
    X_v = np.array(X)
    dot = np.dot(X_v.transpose(),X_v)
    n = np.linalg.norm(X_v,axis=0)
    nn = np.dot(n.reshape(-1,1),n.reshape(1,-1))
    cos_similarity = dot/nn
    return cos_similarity

def corr_sim(X):
    X_v = np.array(X)
    corr = np.corrcoef(X_v.transpose())
    return corr

def sim_feature(feature_X):
    feature = feature_X.columns
    X = feature_X.values
    corr = corr_sim(X)
    feature_corr = pd.DataFrame(corr,index = feature,columns = feature)
    return feature_corr

def get_feature_nan_and_ks(feature_X,Y):
    feature_all = feature_X.columns
    ks_list = []
    miss_ratio_list = []
    for feature in feature_all:
        miss_ratio_list.append(round(feature_X[feature].isnull().sum()*1.0/len(feature_X),5))
        X_y = pd.DataFrame(zip(feature_X[feature],Y)).dropna().values
        ks_list.append(round(ks(X_y[:,1],X_y[:,0])['ks'],5))
    nan_dict = dict(zip(feature_all,miss_ratio_list))
    ks_dict = dict(zip(feature_all,ks_list))
    return nan_dict,ks_dict
        


def select_feature(feature_X,Y,threshold):
    # nan_dict,ks_dict =get_feature_nan_and_ks(feature_X,Y)
    feature_X = feature_X.fillna(-1)
    feature_corr = sim_feature(feature_X)
    feature_all = feature_corr.columns
    select_feature = []
    TF = feature_corr > threshold
    i = 0
    while i < len(feature_all):
        print i
        feature = feature_all[i]
        try:
            above_threshold = feature_corr.loc[feature][TF.loc[feature]].sort_values(ascending=False)
            above_threshold_feature = above_threshold.index.tolist()
            if len(above_threshold) == 1:
                select_feature.append(feature)
                feature_corr = feature_corr.drop(feature,axis=1)
                feature_corr = feature_corr.drop(feature,axis=0)
            else:
                print 'Similarity Result :\n'
                above_threshold = pd.DataFrame(above_threshold.values,index=above_threshold.index,columns=['sim'])
                above_threshold['KS'] = [ks_dict[f] for f in above_threshold.index]
                above_threshold['Miss_Ratio'] = [nan_dict[f] for f in above_threshold.index]
                above_threshold['Num'] = range(len(above_threshold)) 
                print above_threshold
                feature_dict = dict(zip(range(len(above_threshold)),above_threshold.index))
                feature_preserve_num = raw_input('Select what you want to preserve [Input Number or List ,-1 means all , break to stop] : ')
                if feature_preserve_num == 'break':
                    break
                feature_preserve_num = eval(feature_preserve_num)
                if feature_preserve_num == -1:
                    select_feature += above_threshold_feature
                elif isinstance(feature_preserve_num,int) and feature_preserve_num != -1 :
                    try:                        
                        select_feature.append(feature_dict[feature_preserve_num])
                    except:
                        continue
                elif isinstance(feature_preserve_num,list):
                    try:
                        select_feature += [feature_dict[x] for x in feature_preserve_num]
                    except:
                        continue
                else:
                    continue
                feature_corr = feature_corr.drop(above_threshold_feature,axis=1)
                feature_corr = feature_corr.drop(above_threshold_feature,axis=0)
            i = i+1
        except:
            i = i+1
        print select_feature
    return select_feature


feature_X = data[feature_use]
Y = data['label']
nan_dict,ks_dict =get_feature_nan_and_ks(feature_X,Y)                
select_feature_list = select_feature(feature_X,Y,threshold=0.9)
pd.DataFrame(select_feature_list).to_csv('select_feature.csv',index = False,header=False)



def get_top_sim(X,way='corr',top=10):
    feature = X.columns
    sim_feature_list = []
    X_v = X.values
    if way == 'corr':        
        sim = corr_sim(X_v)
    elif way == 'cos':
        sim = cos_sim(X_v)
    order = sim.argsort()[:,::-1]
    order_reverse = sim.argsort()
    for i,f in enumerate(feature):
        sim_feature = feature[order[i,0:top]]
        sim_value = sim[i,:][order[i,0:top]]
        sim_result = [str(v) + '|' +f for f,v in zip(sim_feature,sim_value)]
 
        sim_feature_ = feature[order_reverse[i,0:top]]
        sim_value_ = sim[i,:][order_reverse[i,0:top]]
        sim_result_ = [str(v) + '|' +f for f,v in zip(sim_feature_,sim_value_)]
        
        sim_feature_list.append(sim_result+sim_result_)
    top_sim = np.hstack([np.array(feature).reshape(-1,1),np.array(sim_feature_list)])
    columns = ['target'] + ['top_sim_' + str(i) for i in range(top)] + ['neg_top_sim_' + str(i) for i in range(top)]
    feature_sim_pd = pd.DataFrame(top_sim,columns = columns)
    return feature_sim_pd

