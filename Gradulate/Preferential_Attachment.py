# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:43:51 2016

@author: Admin
"""

from Get_Relation import *
from Get_Data import *
from Time_Tool import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

Start = input('Start Time (year/month):    ')
End = input('Start Time (year/month):    ')

Data = Get_Data(Start,End)
ALL_Relation = Get_Relation(Data,0,1)

# Remove Duplicate
ALL_Relation_RD = ALL_Relation.drop_duplicates(['P','R'])

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
        k_v = 1.0 / (IDs_Dict_Time[P] - T_P)
        if k_v >50:
            k_v=50
        IDs_Dict_V[P] += k_v

    if IDs_Dict_Degree[R] > 1:
        k_v = 1.0 / (IDs_Dict_Time[R] - T_R)
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
b = np.mean(Ydata_log) - k * np.mean(Xdata_log)

x = Xdata
def fun(x,k,b):
    return math.exp(b) * math.pow(x,k)
y =[fun(i,k,b) for i in x]

plt.loglog(Degree_V_Mean.Degree,Degree_V_Mean.V,'o',alpha=0.8,markersize = 3)
plt.loglog(x,y,'k--')
plt.xlabel('K',fontsize = 15)
plt.ylabel(r'$\pi\left(k\right)$',fontsize = 20)
plt.title('Measuring Preferential Attachment' , fontsize  =13)

#Degree_V.to_csv('Degree_V.txt',encoding='utf8',header=True,index = False)

#Degree_V_Mean.to_csv('Degree_V_Mean.txt',encoding='utf8',header=True,index = False)
