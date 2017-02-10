# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:13:33 2016

@author: Admin
"""

import pandas as pd
import re
import glob

def Get_Data(Start,End):

    Files = glob.glob('D:\Graduate\Data\*.txt')
    pattern = r'D:\\Graduate\\Data\\(\d+)'
    Files = sorted(Files , key = lambda x : int(re.findall(pattern , x)[0]))

    Time = []
    for year in range(2006,2017):
        for month in range(1,13):
            Time.append(str(year) + '/' + str(month))
    Time_columns = Time[3:-10]
    Time_Index_Dict = dict(zip(Time_columns,range(len(Time_columns))))
    
    Temp = pd.read_csv(Files[0])
    Columns_Name = Temp.columns 
    
    Data = pd.DataFrame(columns = Columns_Name)
    
    for f in Files[Time_Index_Dict[Start]:Time_Index_Dict[End]+1]:
        Data_Temp = pd.read_csv(f)
        Data = pd.concat([Data,Data_Temp])
        Data.dropna()
        del Data_Temp
        
    return Data