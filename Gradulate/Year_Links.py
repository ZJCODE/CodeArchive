# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:02:21 2016

@author: Admin
"""


from Get_Relation import *
from Get_Data import *
from Time_Tool import *
from PowLaw import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from Tools import *

Data = Get_Data('2006/4','2007/3')
Data.index = range(len(Data))
Relation = Get_Relation(Data,0,0)
Relation = Relation.dropna()

Save_DataFrame(Relation,'2012')

'''
def Get_People_Num(Relation):
    P = set(Relation.ID)|set(Relation.Post_ID)
    return len(P)
 
Active.append(Get_People_Num(Data))

#Relation.columns = [['Source','Target']]
#Save_DataFrame(Relation,'Relation_2008')

Hour = []
for time in Data.Reply_Time:
    hour,minute,second = time.split(' ')[1].split(':')
    hour = int(hour)
    Hour.append(hour)

Month = []
for time in Data.Reply_Time:
    year,month,day = time.split(' ')[0].split('/')
    month = int(month)
    Month.append(month)
    

fig, axes = plt.subplots(2, 1, figsize=(10, 9))   # 构建绘图框架

fig.subplots_adjust(hspace=0.2, wspace=0.3)

from collections import Counter
plt.subplot(211)

CH = Counter(Hour)
plt.plot(CH.keys(),CH.values(),'--s',color='k',alpha=0.8)
plt.xlim([-1,24])
for i in range(len(CH)):    
    plt.text(CH.keys()[i],CH.values()[i]+12000,CH.keys()[i])
for i in range(len(CH)):
    plt.plot([CH.keys()[i],CH.keys()[i]],[0,CH.values()[i]],'k',linewidth=1,alpha=0.8)
plt.xlabel('Hour',fontsize=13)
plt.ylabel('The Amount of Activity',fontsize=13)
ax=plt.gca()
ax.set_xticklabels((' '))  

plt.subplot(212)

CM = Counter(Month)
plt.plot(CM.keys(),CM.values(),'--s',color='k',alpha=0.8)
plt.xlim([0,13])
for i in range(len(CM)):    
    plt.text(CM.keys()[i],CM.values()[i]+9000,CM.keys()[i])
for i in range(len(CM)):
    plt.plot([CM.keys()[i],CM.keys()[i]],[50000,CM.values()[i]],'k',linewidth=1,alpha=0.8)
plt.xlabel('Month',fontsize=13)
plt.ylabel('The Amount of Activity',fontsize=13)
ax=plt.gca()
ax.set_xticklabels((' '))  
'''


Xdata = DS.Degree
Ydata = DS.Strength

Xdata_log = np.log(Xdata) #- np.mean(np.log(Xdata))
Ydata_log = np.log(Ydata) #- np.mean(np.log(Ydata))



 
#Xdata_log = np.log(X_Log_Bin)
#Ydata_log = np.log(Y_Log_Bin)
 
k = float(np.dot(Xdata_log,Ydata_log)) / np.dot(Xdata_log,Xdata_log)
b = np.mean(Ydata_log) - k * np.mean(Xdata_log)

	
x = Xdata
def fun(x,k,b):
    return math.exp(b) * math.pow(x,k)
y =[fun(i,k,b) for i in x]
      
plt.loglog(Xdata,Ydata,'ko',alpha = 0.2,markersize=3)
plt.loglog(x,y,'k--')
plt.title('Degree-Strength Relation',fontsize = 13)
plt.xlabel('k',fontsize = 13)
plt.ylabel('S(k)',fontsize = 13)