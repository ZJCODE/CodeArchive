# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:30:51 2016

@author: Admin
"""

def Generailze_Time(Time):
    Day_Month = [31,28,31,30,31,30,31,31,30,31,30,31]
    year,month,day = Time.split(' ')[0].split('/')
    hour,minute,second = Time.split(' ')[1].split(':')
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    t = minute + 60 * hour + 24 * 60 * day + 24 * 60 * sum(Day_Month[:month - 1])  + 365 * 24 * 60 * (year - 2006)
    return t
    
def Time_Bin(List,Unit):
    #Unit Minute,Hour,Day,Week,Month
    Minute_M = 1
    Hour_M = 60
    Day_M = Hour_M * 24
    Week_M = Day_M * 7
    Month_M = Day_M * 30
    Dict_Time = dict(zip(['Minute','Hour','Day','Week','Month'],[Minute_M,Hour_M,Day_M,Week_M,Month_M]))
    Bins = int(List[-1] / Dict_Time[Unit])
    Count = [0] * Bins
    j = 0 
    for i in range(Bins):
        while j < len(List) and List[j] <= Dict_Time[Unit] * (i+1):
            j += 1
            Count[i] += 1
    
    return Count
    
