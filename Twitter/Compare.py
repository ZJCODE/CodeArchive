# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:35:22 2016

@author: ZJun
"""

import numpy as np
import pandas as pd

def normalize(x):
    x = np.array(x)
    return (x-x.mean())*1.0 / x.std()
    
Queensland_Flu = pd.read_csv('./Data/Queensland2015.csv')
    
Brisbane = Queensland_Flu.Brisbane.values

