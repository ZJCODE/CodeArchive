#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:54:11 2017

@author: zhangjun
"""

import os.path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PreProcess import Get_Stock_Ts
from Pattern_Toolkit import Extract_Pattern_and_Target_With_Resample
from Pattern_Toolkit import Pattern_Cluster,Profit_Judge
import gc
import seaborn
from Calculate_Toolkit import Cos_Similarity
seaborn.set()


file_path = './Pattern_Numpy/n_clusters_'
n_clusters = input('Enter the number of cluster : ')

combine_pattern_pool = np.load(file_path + str(n_clusters)+ '/'+ str(n_clusters) +'_combine_pattern_pool.npy')
combine_profit_mean_pool = np.load(file_path + str(n_clusters)+ '/'+ str(n_clusters) +'_combine_profit_mean_pool.npy')
combine_profit_std_pool = np.load(file_path + str(n_clusters)+ '/'+ str(n_clusters) +'_combine_profit_std_pool.npy')


def Get_Ts_Matched_Pattern(ts,day,combine_pattern_pool,combine_profit_mean_pool,combine_profit_std_pool,similarity_threshold,top_num):
    ts = ts.dropna()
    pattern_len = combine_pattern_pool.shape[1]
    ts_temp = ts[:day]
    pattern = ts_temp[-pattern_len:].values
    #pattern = pattern/pattern[-1]
    similarity_list = Cos_Similarity(pattern,combine_pattern_pool)

    d_zip = zip(combine_pattern_pool,combine_profit_mean_pool,combine_profit_std_pool,similarity_list)
    d_zip = [x for x in d_zip if x[3]>similarity_threshold]
    d_zip.sort(key=lambda x : x[3],reverse=True)
    if top_num < len(d_zip):
        d_zip = d_zip[:top_num]
    if len(d_zip)>0:    
        combine_pattern_pool_select,combine_profit_mean_pool_select,combine_profit_std_pool_select,similarity_list_select = zip(*d_zip)
        combine_pattern_pool_select = np.array(combine_pattern_pool_select)
        combine_profit_mean_pool_select = np.array(combine_profit_mean_pool_select)
        combine_profit_std_pool_select = np.array(combine_profit_std_pool_select)
        similarity_list_select = np.array(similarity_list_select)
        return pattern,combine_pattern_pool_select,combine_profit_mean_pool_select,combine_profit_std_pool_select,similarity_list_select
    else:
        return pattern,np.nan,np.nan,np.nan,np.nan

# Example

data_df = pd.read_csv('data.csv',parse_dates=['TRADE_DT'])
data_df.index = data_df.TRADE_DT
ts = Get_Stock_Ts(data_df,code,filed='S_DQ_ADJCLOSE')

while True:

    code = raw_input('Code : ')#'600781.SH'
    day = raw_input('Day : ') #'2017-4-1'
    similarity_threshold = input('Similarity Threshold : ') #0.99
    top_num = input('Top Num : ')#5

    pattern,combine_pattern_pool_select,combine_profit_mean_pool_select,combine_profit_std_pool_select,similarity_list_select = Get_Ts_Matched_Pattern(ts,day,combine_pattern_pool,combine_profit_mean_pool,combine_profit_std_pool,similarity_threshold,top_num)

    if ~np.isnan(combine_pattern_pool_select).any():
        print 'Similarity : '
        print similarity_list_select
        print 'Mean : '
        print combine_profit_mean_pool_select.mean(0)
        print 'Std : '
        print combine_profit_std_pool_select.mean(0)

    stop = raw_input('going on ? [y/n] ')
    
    if stop == 'y' or stop == 'Y':
        pass
    else:
        break





