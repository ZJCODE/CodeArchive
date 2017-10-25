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
seaborn.set()





pattern_list = np.load('pattern_list.npy')
target_list = np.load('target_list.npy')

# 模式序列长度
pattern_len = pattern_list.shape[1]
# 模式后序序列长度
target_len = target_list.shape[1] - 1

while True:
    os.system('clear')
    # 想要聚类的数量
    n_clusters = input('Enter the number of cluster : ')
    print 'Clustering'
    # 聚类过程
    X,label = Pattern_Cluster(pattern_list,n_clusters = n_clusters,methods = 'K-Means')
    print 'Clustering Finished'
    # 风险收益的阈值
    risk_profit_threshold = input('Enter Risk Profit Threshold : ')
    # 记录每个类别中对象的数目
    label_counter = Counter(label)   
    
    
    # 构建文件目录，用于输出图像
    file_path = './Pic'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    file_path = './Pic/n_clusters_'+str(n_clusters) 
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    file_path += '/risk_profit_threshold_'+str(risk_profit_threshold)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    combine_pattern_pool = np.ones(pattern_len)
    combine_profit_mean_pool = np.ones(target_len)
    combine_profit_std_pool = np.ones(target_len)
    
    print 'Start Draw Pic'
    
    plt.rc('figure',figsize=[12,8])
    
    for cluster_name in range(n_clusters):
        # 计算每个类别后序收益序列，收益均值、标准差，风险收益
        target_list_profit,profit_mean,profit_std,risk_profit = Profit_Judge(target_list,cluster_name,label)
        # 记录该cluster中原始模式的数量
        line_in_cluster = label_counter[cluster_name]
        # 将同一个cluster中的所有模式序列合并
        combine_pattern = X[label==cluster_name].mean(0)
    
        combine_pattern_pool = np.vstack([combine_pattern_pool,combine_pattern])
        combine_profit_mean_pool = np.vstack([combine_profit_mean_pool,profit_mean])
        combine_profit_std_pool = np.vstack([combine_profit_std_pool,profit_std])
        
        # 将符合风险收益阈值的模式绘图保存
        if risk_profit[0] > risk_profit_threshold or risk_profit[1] > risk_profit_threshold :
                
           
            plt.figure()
            plt.subplot(221)
            plt.plot(X[label==cluster_name].mean(0))
            plt.title('Pattern with %d line combined '%(line_in_cluster))
            
            plt.subplot(222)
            plt.plot(profit_mean)
            plt.title('Future Profit Mean')
            
            plt.subplot(223)
            plt.plot(profit_std)
            plt.title('Future Profit Std')
            
            plt.subplot(224)
            plt.plot(risk_profit)
            plt.title('Future Risk Profit')
            
            plt.savefig(file_path + '/[ '+ str(round(risk_profit[0],3)) +' ]_Risk_Profit_Positive Clsuter %s'%(cluster_name)+'.png')
            plt.close('all')

        if risk_profit[0] < -1*risk_profit_threshold :
            plt.figure()
            plt.subplot(221)
            plt.plot(X[label==cluster_name].mean(0))
            plt.title('Pattern with %d line combined '%(line_in_cluster))
            
            plt.subplot(222)
            plt.plot(profit_mean)
            plt.title('Future Profit Mean')
            
            plt.subplot(223)
            plt.plot(profit_std)
            plt.title('Future Profit Std')
            
            plt.subplot(224)
            plt.plot(risk_profit)
            plt.title('Future Risk Profit')
            
            plt.savefig(file_path + '/[ '+ str(round(risk_profit[0],3)) +' ]_Risk_Profit_Negative Clsuter %s '%(cluster_name)+'.png')
            plt.close('all')
    # 将各个模式和各个模式的后序表现存档
    print 'Store Pattern_Info '
    
    file_path = './Pattern_Numpy'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    file_path = './Pattern_Numpy/n_clusters_'+str(n_clusters) 
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    np.save(file_path + '/'+ str(n_clusters) +'_combine_pattern_pool.npy',combine_pattern_pool)
    np.save(file_path + '/'+ str(n_clusters) +'_combine_profit_mean_pool.npy',combine_profit_mean_pool)
    np.save(file_path + '/'+ str(n_clusters) +'_combine_profit_std_pool.npy',combine_profit_std_pool)
    
    pd.DataFrame(combine_pattern_pool).to_csv(file_path + '/'+ str(n_clusters) +'_combine_pattern_pool.csv',index=False,header=False)
    pd.DataFrame(combine_profit_mean_pool).to_csv(file_path + '/'+ str(n_clusters) +'_combine_profit_mean_pool.csv',index=False,header=False)
    pd.DataFrame(combine_profit_std_pool).to_csv(file_path + '/'+ str(n_clusters) +'_combine_profit_std_pool.csv',index=False,header=False)


    # 以上三个文件用于后序模式匹配评判而不是原始序列库，可加速计算
    
    stop = raw_input('try another clustering num ? [y/n] ')
    
    if stop == 'y' or stop == 'Y':
        pass
    else:
        break