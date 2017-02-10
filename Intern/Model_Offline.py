# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:46:00 2016

@author: Admin
"""

import pandas as pd
from datetime import datetime
import numpy as np

def Save_List(List,Name):
    File = Name + '.txt'
    pd.DataFrame({Name:List}).to_csv(File,encoding='utf8',header=True,index = False)
   
import matplotlib.pyplot as plt

def Set_None_Zero(L):
    List = []
    for i in L:
        if np.isnan(i):
            List.append(0)
        else:
            List.append(i)
    return List

def Get_NFS_Normal(ydata,threshold):
    import numpy as np
    from scipy.optimize import curve_fit
    def func_NFS(x, a, b):
        return a*np.power(x,b)
    ydata = np.array(Set_None_Zero(ydata))
    xdata = np.arange(1,len(ydata)+1)
    popt, pcov = curve_fit(func_NFS, xdata, ydata)
    y_trend = func_NFS(xdata,*popt)
    adjusted_low = y_trend * (1-threshold)
    adjusted_high = y_trend * (1+threshold)
    y_normal = []
    for y,l,h in zip(ydata,adjusted_low,adjusted_high):
        if y < l:
            y_normal.append(l)
        elif y > h:
            y_normal.append(h)
        else:
            y_normal.append(y)
    return y_normal
        

def Get_NSO_Normal(ydata,threshold):
    import numpy as np
    from scipy.optimize import curve_fit
    def func_NSO(x,a,b,c):
        return a+b*x+c*x*x
    ydata = np.array(Set_None_Zero(ydata))
    xdata = np.arange(1,len(ydata)+1)
    popt, pcov = curve_fit(func_NSO, xdata, ydata)
    y_trend = func_NSO(xdata,*popt)
    adjusted_low = y_trend * (1-threshold)
    adjusted_high = y_trend * (1+threshold)
    y_normal = []
    for y,l,h in zip(ydata,adjusted_low,adjusted_high):
        if y < l:
            y_normal.append(l)
        elif y > h:
            y_normal.append(h)
        else:
            y_normal.append(y)
    return y_normal





def Get_Base_Uplift(Data,Daily_Data,Channel,year):
    
    if Channel == 'NFS':
        Normalline = Get_NFS_Normal(Data.AMOUNT,0.1)
    else:
        Normalline = Get_NSO_Normal(Data.AMOUNT,0.025)
     
       
    Result = pd.DataFrame({'Week_Start':Data.Week_Start,'Normalline':Normalline})    
    #--------------------------------------------------------------------
    
    #Daily_Data = pd.read_excel('Apply_To_Daily.xlsx')
    if year == 2015:
        
        TS_Week = pd.Series(Data.AMOUNT.values,index = Data.Week_Start)
        
        TS = pd.Series(Daily_Data.AMOUNT.values,index = Daily_Data.Date)
        
        New_TS = TS[datetime(2015,10,8):datetime(2015,12,31)]
        
        if Channel == 'NSO':
            New_TS = New_TS.drop(datetime(2015,12,12))
    
        Week = [date.weekday() for date in New_TS.index]
        TS_DF = pd.DataFrame({'Date':New_TS.index,'AMOUNT':New_TS.values,'Week':Week})
        
        Week_Pivot_Table = TS_DF.pivot_table('AMOUNT','Week',aggfunc = 'mean')
        Percent_Week = Week_Pivot_Table/sum(Week_Pivot_Table)
        
        Percent_Week_Dict = dict(Percent_Week)
        TS_Week_Dict = dict(zip(Result.Week_Start,Result.Normalline))
        
        Start_Time = datetime(2015,9,28)
        detla = datetime(2015,9,28) - datetime(2015,9,27)
        
        TS_Week[Start_Time]
        
        Date_Range = pd.date_range(Start_Time,datetime(2015,12,31),freq='W-MON')
        
        Normal_Sales = []
        Daily_Index = []
        
        for week_start in Date_Range:
            day = week_start
            for i in range(7):
                Daily_Index.append(day.date())
                W= day.weekday()
                Normal_Sales.append(TS_Week_Dict[week_start]*Percent_Week_Dict[W])
                day = day + detla
        New_TS = TS[datetime(2015,10,1):datetime(2015,12,31)]
        Daily_Result = pd.DataFrame({'Date':Daily_Index[3:-3],'Actual Sales':New_TS.values,'Normal_Sales':Normal_Sales[3:-3]})
        
    else:
        TS_Week = pd.Series(Data.AMOUNT.values,index = Data.Week_Start)
        
        TS = pd.Series(Daily_Data.AMOUNT.values,index = Daily_Data.Date)
        
        New_TS = TS[datetime(2014,10,8):datetime(2014,12,31)]
        if Channel == 'NSO':
            New_TS = New_TS.drop(datetime(2014,12,12))
            
        Week = [date.weekday() for date in New_TS.index]
        TS_DF = pd.DataFrame({'Date':New_TS.index,'AMOUNT':New_TS.values,'Week':Week})
        
        Week_Pivot_Table = TS_DF.pivot_table('AMOUNT','Week',aggfunc = 'mean')
        Percent_Week = Week_Pivot_Table/sum(Week_Pivot_Table)
        
        Percent_Week_Dict = dict(Percent_Week)
        TS_Week_Dict = dict(zip(Result.Week_Start,Result.Normalline))
        
        Start_Time = datetime(2014,9,29)
        detla = datetime(2015,9,28) - datetime(2015,9,27)
        
        TS_Week[Start_Time]
        
        Date_Range = pd.date_range(Start_Time,datetime(2014,12,31),freq='W-MON')
        
        Normal_Sales = []
        Daily_Index = []
        
        for week_start in Date_Range:
            day = week_start
            for i in range(7):
                Daily_Index.append(day.date())
                W= day.weekday()
                Normal_Sales.append(TS_Week_Dict[week_start]*Percent_Week_Dict[W])
                day = day + detla
        
        New_TS = TS[datetime(2014,10,1):datetime(2014,12,31)]
        Daily_Result = pd.DataFrame({'Date':Daily_Index[2:-4],'Actual_Sales':New_TS.values,'Normal_Sales':Normal_Sales[2:-4]})    
        Increment = Daily_Result.Actual_Sales - Daily_Result.Normal_Sales
        Daily_Result['Increment'] = Increment
        Uplift_Percent = Daily_Result.Increment / Daily_Result.Normal_Sales
        Daily_Result['Uplift_Percent'] = Uplift_Percent
    return Daily_Result
    
    
#--------2014------------------------------------------------------------------------------------------------    
    
Data = pd.read_excel('2014_NFS_Test.xlsx')
Daily_Data = pd.read_excel('2014_Apply_To_Daily.xlsx')

NFS_Daily_Split = pd.read_excel('NFS_Day_Split_All.xlsx')
NFS_Week_Split = pd.read_excel('NFS_Week_Split_All.xlsx')

NFS_Col = NFS_Daily_Split.columns

for i in range(1,len(NFS_Col)):
    TS_Daily = pd.Series(NFS_Daily_Split.ix[:,i].values,index = NFS_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2014,10,1):datetime(2015,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NFS_Week_Split.ix[:,i].values,index = NFS_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2014,3,3):datetime(2015,3,23)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,'NFS',2014)
#    Daily_Result.to_excel('.//NFS//2014_NFS_'+str(i)+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    
    Daily_Result.to_excel('.//NFS//2014_NFS_'+NFS_Col[i]+'.xlsx',encoding='gb2312',header=True,index = False)    

#---------------------------------------------------------------------

Data = pd.read_excel('2014_NSO_Test.xlsx')
Daily_Data = pd.read_excel('2014_Apply_To_Daily.xlsx')

NSO_Daily_Split = pd.read_excel('NSO_Day_Split_All.xlsx')
NSO_Week_Split = pd.read_excel('NSO_Week_Split_All.xlsx')

NSO_Col = NSO_Daily_Split.columns

for i in range(1,len(NSO_Col)):
    TS_Daily = pd.Series(NSO_Daily_Split.ix[:,i].values,index = NSO_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2014,10,1):datetime(2015,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NSO_Week_Split.ix[:,i].values,index = NSO_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2014,3,3):datetime(2015,3,23)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,'NSO',2014)
    Daily_Result.to_excel('.//NSO//2014_NSO_'+NSO_Col[i]+'.xlsx',encoding='gb2312',header=True,index = False)    


#-------2015----------------------------------------------------------------------------------------------------------------

Data = pd.read_excel('2015_NFS_Test.xlsx')
Daily_Data = pd.read_excel('2015_Apply_To_Daily.xlsx')

NFS_Daily_Split = pd.read_excel('NFS_Day_Split_All.xlsx')
NFS_Week_Split = pd.read_excel('NFS_Week_Split_All.xlsx')

NFS_Col = NFS_Daily_Split.columns

for i in range(1,len(NFS_Col)):
    TS_Daily = pd.Series(NFS_Daily_Split.ix[:,i].values,index = NFS_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2015,10,1):datetime(2016,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NFS_Week_Split.ix[:,i].values,index = NFS_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2015,3,2):datetime(2016,3,21)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,'NFS',2015)
#    Daily_Result.to_excel('.//NFS//2015_NFS_'+str(i)+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    
    Daily_Result.to_excel('.//NFS//2015_NFS_'+NFS_Col[i]+'.xlsx',encoding='gb2312',header=True,index = False)    

#---------------------------------------------------------------------

Data = pd.read_excel('2015_NSO_Test.xlsx')
Daily_Data = pd.read_excel('2015_Apply_To_Daily.xlsx')

NSO_Daily_Split = pd.read_excel('NSO_Day_Split_All.xlsx')
NSO_Week_Split = pd.read_excel('NSO_Week_Split_All.xlsx')

NSO_Col = NSO_Daily_Split.columns

for i in range(1,len(NSO_Col)):
    TS_Daily = pd.Series(NSO_Daily_Split.ix[:,i].values,index = NSO_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2015,10,1):datetime(2016,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NSO_Week_Split.ix[:,i].values,index = NSO_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2015,3,2):datetime(2016,3,21)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,'NSO',2015)
    Daily_Result.to_excel('.//NSO//2015_NSO_'+NSO_Col[i]+'.xlsx',encoding='gb2312',header=True,index = False)  