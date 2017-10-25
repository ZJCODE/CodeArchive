#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:10:21 2017

@author: zhangjun
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from Calculate_Toolkit import Cos_Similarity,Manhattan_Distance_Similarity,Euclidean_Distance_Similarity


def Transfer_Array_To_Discrete_Pattern(x,n_level=20,to_str = False):
    '''
    将序列转变为归一化后的离散形式

    n_level : 划分粒度

    -----------------------------------
    Example 

    x = [3,4,5,6,4,6,6,13,4,9,8,8,7,3]

    n_level = 10

    Transfer_To_Discrete_Pattern(x,n_level=10,to_str = False) : 

        array([0, 0, 1, 2, 0, 2, 2, 9, 0, 5, 4, 4, 3, 0]) # 0-9之间的数值

    Transfer_To_Discrete_Pattern(x,n_level=10,to_str = True) : 

        '0-0-1-2-0-2-2-9-0-5-4-4-3-0'

    -----------------------------------

    '''
    x = np.array(x).astype('float')
    min_v = min(x)
    max_v = max(x)
    if max_v > min_v:    
        x_0_1 = (x-min_v)/(max_v-min_v)-10e-10
        x_discrete = np.array([int(round(i)) for i in x_0_1 * n_level])
    else:
        x_discrete = np.ones(len(x)).astype('int')

    if to_str == True:
        x_discrete = '-'.join([str(i) for i in x_discrete])
    return x_discrete

def Transfer_Matrix_To_Discrete_Pattern(M):
    '''
    
    将矩阵的每一行转变为归一化后的离散形式

    -----------------------------------
    Example     

    M

    array([[ 0,  8,  8,  9,  3,  6],
           [ 6,  7,  4,  8,  6,  1],
           [ 8,  9,  6, 10,  8,  3],
           [ 1,  1,  9,  5,  5,  3],
           [ 8,  4,  2,  1,  8,  4]])
    
    Transfer_Matrix_To_Discrete_Pattern(M)

    array([[ 0, 18, 18, 20,  7, 13],
           [14, 17,  9, 20, 14,  0],
           [14, 17,  9, 20, 14,  0],
           [ 0,  0, 20, 10, 10,  5],
           [20,  9,  3,  0, 20,  9]])

    '''
    M_discrete = np.apply_along_axis(Transfer_Array_To_Discrete_Pattern,1,M)

    return M_discrete


def Extract_Pattern_and_Target(ts,pattern_len,target_len,normalization=False):
    '''
    获取历史模式和该模式后的走势序列
    
    -----------------------------------
    Example
    
    ts =
        2017-05-24    236.18
        2017-05-25    238.04
        2017-05-26    235.26
        2017-05-31    236.65
        2017-06-01    235.26
        2017-06-02    230.16
        2017-06-05    231.55
        2017-06-06    234.33
        2017-06-07    244.98
        2017-06-08    253.78
        2017-06-09    249.61
        2017-06-12    242.67
        2017-06-13    244.06
        2017-06-14    243.59
        2017-06-15    245.45
        2017-06-16    244.06
        2017-06-19    245.45
        2017-06-20    244.98
        2017-06-21    245.91
        2017-06-22    244.06
    
    pattern_len =  4
    
    target_len =  2
    
    pattern_list,target_list = Extract_Pattern_and_Target(ts,pattern_len,target_len,normalization=False)

    pattern_list]: 
    
        array([[   1.  ,    1.  ,    1.  ,    1.  ],
               [ 236.18,  238.04,  235.26,  236.65],
               [ 238.04,  235.26,  236.65,  235.26],
               ..., 
               [ 244.06,  243.59,  245.45,  244.06],
               [ 243.59,  245.45,  244.06,  245.45],
               [ 245.45,  244.06,  245.45,  244.98]])

    target_list: 
        
        array([[   1.  ,    1.  ,    1.  ],
               [ 236.65,  235.26,  230.16],
               [ 235.26,  230.16,  231.55],
               ..., 
               [ 244.06,  245.45,  244.98],
               [ 245.45,  244.98,  245.91],
               [ 244.98,  245.91,  244.06]])
    
    
    
    pattern_list,target_list = Extract_Pattern_and_Target(ts,pattern_len,target_len,normalization=True)

    pattern_list: 
        
        array([[ 1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.99801394,  1.00587365,  0.99412635,  1.        ],
               [ 1.01181671,  1.        ,  1.00590836,  1.        ],
               ..., 
               [ 1.        ,  0.99807424,  1.00569532,  1.        ],
               [ 0.99242208,  1.        ,  0.99433693,  1.        ],
               [ 1.00191852,  0.99624459,  1.00191852,  1.        ]])
    
    target_list: 
        
        array([[ 1.        ,  1.        ,  1.        ],
               [ 1.        ,  0.99412635,  0.97257553],
               [ 1.        ,  0.97832186,  0.98423021],
               ..., 
               [ 1.        ,  1.00569532,  1.00376956],
               [ 1.        ,  0.99808515,  1.00187411],
               [ 1.        ,  1.00379623,  0.99624459]])
        
    -----------------------------------
    '''

    ts = ts.dropna()
    
    length = pattern_len + target_len
    pattern_list = np.ones(pattern_len)
    target_list = np.ones(target_len+1)
    for i in range(len(ts)-length+1):
        try:        
            pattern = ts[i:i+pattern_len].values
            # pattern 的最后一天和 target 的第一天重合，作为之后的参照
            target = ts[i+pattern_len-1:i+pattern_len+target_len].values  
        except:
            pattern = ts[i:i+pattern_len]
            # pattern 的最后一天和 target 的第一天重合，作为之后的参照
            target = ts[i+pattern_len-1:i+pattern_len+target_len]
        # 分界点数值归一到1
        if normalization == True:            
            pattern = pattern/pattern[-1]
            target = target/target[0]
        # 如果序列中存在缺失值就丢弃该序列
        if (~np.isnan(pattern).any()) and (~np.isnan(target).any()) and  (len(pattern) == pattern_len) and (len(target) == target_len+1) :            
            pattern_list = np.vstack([pattern_list,pattern])
            target_list = np.vstack([target_list,target])
    if len(pattern_list.shape) == 2:    
        pattern_list = pattern_list[1:,:]
        target_list = target_list[1:,:]
    return pattern_list,target_list



def Resample_Ts(ts,resample_ratio):
    '''
    对时间序列进行rolling mean（考虑偏移）
    '''
    ts_resample_list = []
    for i in range(resample_ratio):    
        ts_resample = ts[i:].rolling(resample_ratio).mean()
        ts_resample_list.append(ts_resample)
    return ts_resample_list


def Extract_Pattern_and_Target_With_Resample(ts,pattern_len,target_len,resample_list,normalization=False):
    '''
    获取历史模式和该模式后的走势序列(考虑rolling)
    '''

    # 初始化模式库
    pattern_list_all = np.ones(pattern_len)
    target_list_all = np.ones(target_len+1)
    # 对时间序列按照 freq 个 resample
    pattern_list,target_list = Extract_Pattern_and_Target(ts,pattern_len,target_len,normalization)
    pattern_list_all = np.vstack([pattern_list_all,pattern_list])
    target_list_all = np.vstack([target_list_all,target_list])
    # 对序列进一步缩放，比率依据 resample_list 中的数
    for resample in resample_list:
        ts_resample_list = Resample_Ts(ts,resample)
        for ts in ts_resample_list:        
            pattern_list,target_list = Extract_Pattern_and_Target(ts,pattern_len,target_len,normalization)
            pattern_list_all = np.vstack([pattern_list_all,pattern_list])
            target_list_all = np.vstack([target_list_all,target_list])
    return pattern_list_all,target_list_all


def Pattern_Cluster(pattern_list,n_clusters=250,methods = 'K-Means'):
    '''
    对模式序列进行聚类，返回模式和聚类标签
    '''
    X = pattern_list
    print 'Shape  : ' + str(pattern_list.shape)
    if methods == 'K-Means':   # 较快，效果也还不错
        batch_size = int(len(X)/20)
        mbk = MiniBatchKMeans(init='k-means++',n_clusters = n_clusters , batch_size=batch_size)
        mbk.fit(X)
        label = mbk.labels_
    elif methods == 'Mean-Shift': # 有些慢
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=2000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        label = ms.labels_
    elif methods == 'Agglomerative': # 有些慢
        clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
        clustering.fit(X)
        label = clustering.labels_
    return X,label



def Profit_Judge(target_list,cluster_name,label):
    '''
    对预期收益进行评估

    target_list : 模式的后结序列，与pattern_list相互对应
    cluster_name : cluster的序号
    label : cluster 算法输出的分类标签序列
    '''
    
    target_list_profit  = (target_list - target_list[:,0][:,np.newaxis])/target_list[:,0][:,np.newaxis]
    profit_mean = np.mean(target_list_profit[label==cluster_name],axis=0)[1:]
    profit_std = np.std(target_list_profit[label==cluster_name],axis=0)[1:]
    risk_profit = profit_mean/profit_std # 收益／标准差

    return target_list_profit,profit_mean,profit_std,risk_profit

