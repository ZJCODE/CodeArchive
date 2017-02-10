#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:33:38 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
Data = pd.read_csv('syzsid.csv')
Data['YHID'] = [int(a) for a in Data.YHID]
Data['CONS_NO'] = [str.upper(a) for a in Data.CONS_NO]
Nan_Data = Data[~ (Data.sfqd < 2)]
No_Nan_Data = Data[Data.sfqd < 2]

X  = No_Nan_Data.YHID.values
y = No_Nan_Data.sfqd.values
X =X[:,np.newaxis]
X_ = Nan_Data.YHID.values
X_ = X_[:,np.newaxis]

from sklearn import neighbors 

knn = neighbors.KNeighborsClassifier(n_neighbors=2)

knn.fit(X,y)

y_ = knn.predict(X_)

Nan_Data['sfqd'] = y_

Modify_Data = pd.concat([No_Nan_Data,Nan_Data])


def Get_Sub_DF(DF,col,sub_list):
    Sub_DF = DF[DF[col].isin(sub_list)]
    return Sub_DF
    
f = open('test.csv','r')

test = []

for line in f.readlines():
    test.append(line.strip())
    
Sub = Get_Sub_DF(Modify_Data,'CONS_NO',test)

Sub_sort = Sub.sort_values('sfqd',ascending=False)

Sub_sort.index = range(len(Sub_sort))

Sub_sort.CONS_NO.to_csv('Submmision.csv',index=False,header=False)