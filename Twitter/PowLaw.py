# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:53:13 2016

@author: Admin
"""

# coding: utf-8

def Generate_PowLaw_Data(gamma,Len):
    import math
    def fun(x,gamma):
        return math.pow(x,-gamma)
    Data_F = [fun(i,gamma) for i in range(1,Len+1)]
    F_min = min(Data_F)
    Data_Max = int(1 / F_min) +1
    Data = [i * Data_Max for i in Data_F]
    return Data
	
 
def Log_Bin(X,c):
    import math
    N = int (math.log((c-1) * len(X) + 1) / float(math.log(c)) )
    X_Log_Bin = []
    X_Index_Bin = []
    X_Index = range(1,len(X)+1)
    Sum_X = float(sum(X))
    Index = 0 
    for i in range(N):
        Len = int (math.pow(c,i))
        X_Log_Bin.append(sum(X[Index:Index+Len])/Len/Sum_X)
        X_Index_Bin.append(sum(X_Index[Index:Index+Len])/(Len))
        Index = Index + Len
    #X_Log_Bin.append(sum(X[Index+Len:])/(len(X)-Index-Len)/Sum_X)
    #X_Index_Bin.append(sum(X_Index[Index+Len:])/(len(X)-Index-Len))
    return X_Index_Bin , X_Log_Bin
		


def PowLawFit(X,Draw=0):
    import matplotlib.pyplot as plt
    import math
    import numpy as np


    X_index = range(1,len(X)+1)
    Pair = zip(X_index,X)
    NoZeroPair = [i for i in Pair if i[1] != 0]
    Xdata,Ydata = zip(*NoZeroPair)
    Sum = sum(Ydata)
    Ydata = [float(i) / Sum for i in Ydata]
	
		
    Xdata_log = np.log(Xdata) - np.mean(np.log(Xdata))
    Ydata_log = np.log(Ydata) - np.mean(np.log(Ydata))


    X_Log_Bin , Y_Log_Bin = Log_Bin(X,1.5)
    BinPair = zip(X_Log_Bin , Y_Log_Bin)
    BinNoZeroPair = [i for i in BinPair if i[1] != 0]
    X_Log_Bin , Y_Log_Bin = zip(*BinNoZeroPair)
    X_Log_Bin_log = np.log(X_Log_Bin)- np.mean(np.log(X_Log_Bin))
    Y_Log_Bin_log = np.log(Y_Log_Bin)- np.mean(np.log(Y_Log_Bin))
     
 
    k = float(np.dot(Xdata_log,Ydata_log)) / np.dot(Xdata_log,Xdata_log)
    b = np.mean(Ydata_log) - k * np.mean(Xdata_log)
    
    k_test = float(np.dot(X_Log_Bin_log,Y_Log_Bin_log)) / np.dot(X_Log_Bin_log,X_Log_Bin_log)
    b_text= np.mean(Y_Log_Bin_log) - k * np.mean(X_Log_Bin_log)
    
    x = X_index
    def fun(x,k,b):
        return math.exp(b) * math.pow(x,k)
    y =[fun(i,k_test,b_text) for i in x]
    if Draw == 1:            
        plt.cla()
        plt.loglog(Xdata,Ydata,'go',alpha = 0.2,markersize=5)
        plt.loglog(x,y,'k--')
        plt.loglog(X_Log_Bin,Y_Log_Bin,'rs',markersize=7)
        title ='Degree Distribution' + '  k = ' + str(-k_test)
        plt.title(title)
        plt.xlabel('D')
        plt.ylabel('P(D)')


    return -k , -k_test
 


def Historgam_List(Data,unit):
    List = sorted(Data)
    Bins = int(List[-1] / unit) + 1
    Count = [0] * Bins
    j = 0 
    for i in range(Bins):
        while j < len(List) and List[j] < unit * (i+1):
            j += 1
            Count[i] += 1
    
    return Count
    