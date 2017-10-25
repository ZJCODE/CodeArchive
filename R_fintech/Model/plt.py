# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:14:59 2017

@author: zhangjun
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pred = np.load('pred_1506482659.npy')

for i,s in enumerate(pred[0]):
    if s in ['fqgj','zzjr_55695']:
        continue
    num = i+1
    sns.kdeplot(pred[num],label=s)
