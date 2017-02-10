# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 16:24:27 2016

@author: Admin
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt

from datetime import datetime


#glob.glob('.\Data\*.xlsx')

Nike_Online = pd.read_excel('.\\Data\\Nike_Online.xlsx')
Tmall_Online = pd.read_excel('.\\Data\\Tmall_Online.xlsx')
NFS_Offline = pd.read_excel('.\\Data\\Offline_Data_NFS.xlsx')
NSO_Offline = pd.read_excel('.\\Data\\Offline_Data_NSO.xlsx')

plt.rc('figure', figsize=(10, 5))

#-------------------------------------------------------------------



#创建时间序列
TS_Nike = pd.Series(Nike_Online.AMOUNT_SUM.values,index = Nike_Online.Date)
TS_Tmall = pd.Series(Tmall_Online.AMOUNT_SUM.values,index = Tmall_Online.Date)
TS_NFS = NFS_Offline.pivot_table('AMOUNT','Date',aggfunc='sum')
TS_NSO = NSO_Offline.pivot_table('AMOUNT','Date',aggfunc='sum')
#------------------------------------------------------


#双11销售额的比例
TS_Nike[datetime(2014,11,11)]/sum(TS[datetime(2014,1,1):datetime(2014,12,31)])
#------------------------------------------------

#绘制阶段图
Start_Time = datetime(2015,1,1)
End_Time = datetime(2015,12,31)

def Draw_TS(TS,Start_Time = min(TS.index),End_Time = max(TS.index)):
    TS_P = TS[Start_Time:End_Time]
    plt.plot(TS_P.index,TS_P,'-o',alpha=0.5,markersize=2)

Draw_TS(TS_Nike,Start_Time,End_Time)
Draw_TS(TS_Tmall,Start_Time,End_Time)
#---------------------------------------


#自相关判定
#plt.scatter(TS_Nike[1:],TS_Nike[:-1])
plt.scatter(TS_Tmall,TS_Tmall.shift(1),alpha=0.5)
plt.xlim([0,2000000])
plt.ylim([0,2000000])
#-------------------------------


#检测打折情况
alpha = 0.5
n=3
Adj_Amount = []
Adj_Amount = Adj_Amount + list(Nike_Online.AMOUNT_SUM[:n])
for i in range(n,len(Nike_Online)):
    if (Nike_Online.AMOUNT_SUM[i]-Nike_Online.AMOUNT_SUM[i-1]) *1.0 /  Nike_Online.AMOUNT_SUM[i-1] > alpha:
        Adj_Amount.append(sum(Adj_Amount[i-n:i])*1.0/n)
    else:
        Adj_Amount.append(Nike_Online.AMOUNT_SUM[i])

plt.plot(Nike_Online.Date,Adj_Amount)
#--------------------------------------------------------