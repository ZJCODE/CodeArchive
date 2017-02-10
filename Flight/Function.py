# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:47:37 2016

@author: ZJun
"""

import pandas as pd


  


def SplitDataByEach10Min(Data,timecolumn):
    Data = Data.dropna(subset = [timecolumn])
    Data = Data.sort_values(by = timecolumn)
    Data.index = range(len(Data))
    TimeSetList = list(set(Data[timecolumn]))
    MinTime = min(TimeSetList)
    MaxTime = max(TimeSetList)

    def Ceil_Time(time):
        from datetime import timedelta
        def Celi_Minute(Minute):
            return int(10*(round(Minute/10)+1))
        try:            
            T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)+ timedelta(0,0,0,0,Celi_Minute(time.minute),0)
        except:
            try:            
                T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)
            except:
                T = time
        return T
    
    TimeRange = pd.date_range(start = Ceil_Time(MinTime) , end = Ceil_Time(MaxTime) , freq = '10Min') # Freq : 10Min , 5H , ....
    
    TimeSplitData = []
    
    Split_Index_After = 0 ; Split_Index_Before = 0 ; TimeRange_Index = 0
    
    while Split_Index_After < len(Data):
        ThresholdTime = TimeRange[TimeRange_Index]
        
        if Data[timecolumn][Split_Index_After] < ThresholdTime:
            Split_Index_After += 1
        else:
            TimeSplitData.append(Data[Split_Index_Before:Split_Index_After]) 
            Split_Index_Before = Split_Index_After
            TimeRange_Index += 1 
            
    return TimeSplitData , TimeRange
 

    