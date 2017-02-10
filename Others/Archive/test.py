#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:50:21 2016

@author: ZJun
"""

import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
Data = pd.read_table('data.txt').ix[:,0].values
plt.plot(Data)

def GetChangePoint(Data):    
    L = []
    for index in range(1,len(Data)-1):
        left = Data[index] - Data[index-1]
        right = Data[index+1] - Data[index]
        flag = left*right
        #print flag
        if flag <= 0 and (Data[index-1] - Data[index+1] != 0) :
            L.append(index)
    return L
    
