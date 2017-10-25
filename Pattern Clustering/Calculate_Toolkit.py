#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:26:29 2017

@author: zhangjun
"""
from __future__ import division
import numpy as np
from collections import Counter

def sigmoid(x):
    '''
    转变数据到 (-1,1)，将正数转变到（0,1）
    
    -----------------------------------
    Example
    
    sigmoid(np.linspace(-10,10,100)): 
        
        array([-0.9999092 , -0.99988888, -0.999864  , ...,  0.999864  ,
        0.99988888,  0.9999092 ])
    
    -----------------------------------
    '''
    return 2*1/(1+np.exp(-x)) -1
    
    
def Cos_Similarity(x,M,norm_M = []):
    '''
    
    计算某一向量与一系列向量的余弦相似性
    
    -----------------------------------
    Example 
    
    x :
        array([2, 3, 4])
    M :
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 2,  3,  4],
               [ 9, 10, 11]])
    
    Cos_Similarity(x,M):
        array([ 0.91350028,0.99792889,  1, 0.98307207])
        
    Cos_Similarity(x,x):
        1.000000
    -----------------------------------
    
    '''
    #M = np.array(M).astype('float')
    #x = np.array(x).astype('float')
    if len(M.shape) == 2:
        if len(norm_M) == 0 :
            cos_similarity = np.dot(M,x.T) / (np.linalg.norm(M,axis=1)*np.linalg.norm(x))
        else:
            cos_similarity = np.dot(M,x.T) / (norm_M*np.linalg.norm(x))
    else:
        cos_similarity = np.dot(M,x.T) / (np.linalg.norm(M)*np.linalg.norm(x))
    return cos_similarity


def Manhattan_Distance_Similarity(x,M,distance=False):
    '''
    
    计算某一向量与一系列向量的基于曼哈顿距离的相似性（修正过）/曼哈顿距离
    
    -----------------------------------
    Example 
    
    x :
        array([2, 3, 4])
    M :
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 2,  3,  4],
               [ 9, 10, 11]])
    
    [Similarity]
    
    Manhattan_Distance_Similarity(x,M,distance=False): 
        array([ 0.875647  ,  0.97500521,  1.        ,  0.9206219 ])

    Manhattan_Distance_Similarity(x,x,distance=False): 
        1.0
        
    [Distance]
    
    Manhattan_Distance_Similarity(x,M,distance=True): 
        array([ 2.,  1.,  0.,  7.])
        
    Manhattan_Distance_Similarity(x,x,distance=True):
        0.0
    -----------------------------------
    
    '''
    #M = np.array(M).astype('float')
    #x = np.array(x).astype('float')
    if distance == False:
        x = x/x[-1]
        if len(M.shape) == 2:       
            M = M / M[:,-1].reshape(-1,1) # 使各个量值大小统一
            manhattan_similarity = 1 - sigmoid((abs(M-x)).mean(1))
        else:
            M = M/M[-1]
            manhattan_similarity = 1 - sigmoid((abs(M-x)).mean())
    else:
        if len(M.shape) == 2:       
            manhattan_similarity = (abs(M-x)).mean(1)
        else:
            manhattan_similarity =(abs(M-x)).mean()
        
    return manhattan_similarity


def Euclidean_Distance_Similarity(x,M,distance = False):
    '''

    计算某一向量与一系列向量的基于欧拉距离的相似性（修正过）/欧拉距离

    -----------------------------------
    Example 
    
    x :
        array([2, 3, 4])
    M :
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 2,  3,  4],
               [ 9, 10, 11]])
    
    [Similarity]
    
    Euclidean_Distance_Similarity(x,M,distance = False):
        array([ 0.72754988,  0.94415646,  1.        ,  0.82398329])
        
    Euclidean_Distance_Similarity(x,x,distance=False): 
        1.0
        
    [Distance]
    
    Euclidean_Distance_Similarity(x,M,distance = True): 
        array([  3.46410162,   1.73205081,   0.        ,  12.12435565])
        
    Euclidean_Distance_Similarity(x,x,distance=True):
        0.0
    -----------------------------------
    
    '''
    #M = np.array(M).astype('float')
    #x = np.array(x).astype('float')
    if distance == False:
        x = x/x[-1]
        if len(M.shape) == 2:
            M = M / M[:,-1].reshape(-1,1) # 使各个量值大小统一
            #euclidean_similarity = 1 - sigmoid(np.sqrt(np.power(M-x,2).sum(1)))
            euclidean_similarity = 1 - sigmoid(np.linalg.norm((M-x),axis=1))
        else:
            M = M / M[-1] # 使各个量值大小统一
            #euclidean_similarity = 1 - sigmoid(np.sqrt(np.power(M-x,2).sum()))
            euclidean_similarity = 1 - sigmoid(np.linalg.norm((M-x)))
    else:
        if len(M.shape) == 2:
            euclidean_similarity = np.sqrt(np.power(M-x,2).sum(1))
        else:
            euclidean_similarity = np.sqrt(np.power(M-x,2).sum())
            
    return euclidean_similarity


def Counter_Sort(x):
    '''
    对数组计数统计，并从大到小排序

    -----------------------------------

    Example

    x = [3,3,3,3,2,2,2,3,5,55,5,4]

    Counter_Sort(x)
        [(3, 5), (2, 3), (5, 2), (4, 1), (55, 1)]

    -----------------------------------
    '''
    c = Counter(x)
    L = list(c.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L


    
