# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 09:40:47 2016

@author: Admin
"""

from datetime import datetime

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()
from datetime import datetime

Data_NFS = pd.read_excel('NFS_Daily.xlsx')
Data_NSO = pd.read_excel('NSO_Daily.xlsx')


def Save_DataFrame(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)


'''
TS_Nike = pd.Series(Nike_Online.AMOUNT_SUM.values,index = Nike_Online.Date)

TS = TS_Nike

Start_Time = datetime(2016,6,13)
End_Time = datetime(2016,6,19)


def Draw_TS(TS,Start_Time = min(TS.index),End_Time = max(TS.index)):
    plt.rc('figure', figsize=(10, 5))
    TS_P = TS[Start_Time:End_Time]
    plt.plot(TS_P.index,TS_P,'-o',alpha=0.5,markersize=2)


plt.scatter(TS_Nike,TS_Nike.shift(1))

#dates = pd.date_range('20160610','20160620')

dates = [datetime(2011,1,1),datetime(2011,1,2),datetime(2011,1,5)]

ts = pd.Series(np.random.randn(3),index=dates)
'''

def Transfer_To_Week(Data):
    def Set_None_Zero(L):
        List = []
        for i in L:
            if np.isnan(i):
                List.append(0)
            else:
                List.append(i)
        return List
        
    detla = datetime(2000,6,7) - datetime(2000,6,1)

    TS_Original = pd.Series(Data.AMOUNT.values,index = Data.Date)
    
    TS_Range = TS_Original.resample('D')
    TS_Range  = TS_Range[pd.date_range('28/9/2015','3/31/2016')]
    TS= pd.Series(Set_None_Zero(TS_Range),index = TS_Range.index)
    
    Start_Time = datetime(2015,9,28)
    TS_Week_Index = []
    TS_Week = []
    for t in TS[Start_Time::7].index:
        
        TS_Week_Index.append(str(t.date()))
        TS_Week.append(sum(TS[t:t+detla]))
    Week_Data  = pd.DataFrame({'Week':TS_Week_Index,'AMOUNT_SUM':TS_Week})
    return Week_Data
    
    
