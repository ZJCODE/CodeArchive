# -*- coding: utf-8 -*-
"""
Created on Wed May 04 21:11:31 2016

@author: Admin
"""

from Time_Tool import *
import numpy as np
import pandas as pd
import math

def Preferential_Attachment(Relation_With_Time):
    
    ALL_Relation_RD = Relation_With_Time#.drop_duplicates(['P','R'])
    
    # Remove Reply Self
    Judge = [i != j for i,j in zip(ALL_Relation_RD.P,ALL_Relation_RD.R)]
    Relation = ALL_Relation_RD[Judge]
    Relation.index = range(len(Relation))
    GTime = [Generailze_Time(time) for time in Relation.Time]
    Relation['GTime'] = GTime
    SRelation = Relation.sort('GTime')
    SRelation.index = range(len(SRelation))
    
    
    IDs = list(set(SRelation.P)|set(SRelation.R))
    
    
    IDs_Dict_Time = dict(zip(IDs,[0]*len(IDs)))
    IDs_Dict_Degree = dict(zip(IDs,[0]*len(IDs)))
    IDs_Dict_V = dict(zip(IDs,[0]*len(IDs)))
    
    
    for P,R,T in zip(SRelation.P,SRelation.R,SRelation.GTime):
        T_P = IDs_Dict_Time[P]
        T_R = IDs_Dict_Time[R]
        D_P = IDs_Dict_Degree[P]
        D_R = IDs_Dict_Degree[R]
        IDs_Dict_Degree[P] = D_P +1 
        IDs_Dict_Degree[R] = D_R +1
        IDs_Dict_Time[P] = T
        IDs_Dict_Time[R] = T
        
        if IDs_Dict_Degree[P] > 1:
            # delat K / delta T
            k_v = IDs_Dict_Degree[P]*1.0 / (IDs_Dict_Time[P] - T_P)
            if k_v >50:
                k_v=50
            IDs_Dict_V[P] += k_v
    
        if IDs_Dict_Degree[R] > 1:
            k_v = IDs_Dict_Degree[R]*1.0 / (IDs_Dict_Time[R] - T_R)
            if k_v >50:
                k_v=50
            IDs_Dict_V[R] += k_v
            
    V = [IDs_Dict_V[id] for id in IDs]
    Degree = [IDs_Dict_Degree[id] for id in IDs]
    
    
    Degree_V = pd.DataFrame({'Degree':Degree,'V':V})
    
    Degree_V_Mean_DF = pd.pivot_table(Degree_V,values = 'V',index = 'Degree' ,aggfunc='mean')
    Degree_V_Mean = pd.DataFrame(Degree_V_Mean_DF)
    Degree_V_Mean['Degree'] = list(Degree_V_Mean.index)
    
    
    
    Xdata = Degree_V_Mean.Degree
    Ydata = Degree_V_Mean.V
    Xdata_log = np.log(Xdata)[1:]
    Ydata_log = np.log(Ydata)[1:]
    
    k = float(np.dot(Xdata_log,Ydata_log)) / np.dot(Xdata_log,Xdata_log)
    
    return k ,Xdata,Ydata
    
k,Xdata,Ydata = Preferential_Attachment(Relation)