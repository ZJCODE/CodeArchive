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
   




def Get_Base_Uplift(Data,Daily_Data,year):
    
    Average_Pivot_Table = Data.pivot_table('AMOUNT','Month',aggfunc = 'mean')
    
    threshold = 0.025
    Month_Index = Average_Pivot_Table.index
    Month_Average = list(Average_Pivot_Table.values)
    Month_UP = [(1+threshold)*a for a  in Month_Average]
    Month_Low = [(1-threshold)*a for a  in Month_Average]
    
    Average_Dict = dict(zip(Month_Index,Month_Average))
    UP_Dict = dict(zip(Month_Index,Month_UP))
    Low_Dict = dict(zip(Month_Index,Month_Low))
    
    BaseLine = []
    
    for month,data,flag in zip(Data.Month,Data.AMOUNT,Data.Flag):
        if month == 10 or month == 11:
            month = 9
        if month == 222:
            month = 111
        if data > Low_Dict[month] and data < UP_Dict[month]:
            BaseLine.append(data)
        else:
            if flag == -1:
                BaseLine.append(Low_Dict[month])
            elif flag == 0:
                BaseLine.append(Average_Dict[month])
            else:
                BaseLine.append(UP_Dict[month])
            
    Result = pd.DataFrame({'Week_Start':Data.Week_Start,'Sales':Data.AMOUNT,'BaseLine':BaseLine})
    Sales_Incremental = Result.Sales - Result.BaseLine
    Sales_Uplift = Sales_Incremental / Result.BaseLine
    Result['Sales_Incremental'] = Sales_Incremental
    Result['Sales_Uplift'] = Sales_Uplift
    
    #--------------------------------------------------------------------
    
    #Daily_Data = pd.read_excel('Apply_To_Daily.xlsx')
    if year == 2015:
        
        TS_Week = pd.Series(Data.AMOUNT.values,index = Data.Week_Start)
        
        TS = pd.Series(Daily_Data.AMOUNT.values,index = Daily_Data.Date)
        
        New_TS = TS[datetime(2015,10,1):datetime(2015,12,31)]
        Week = [date.weekday() for date in New_TS.index]
        TS_DF = pd.DataFrame({'Date':New_TS.index,'AMOUNT':New_TS.values,'Week':Week})
        
        Week_Pivot_Table = TS_DF.pivot_table('AMOUNT','Week',aggfunc = 'mean')
        Percent_Week = Week_Pivot_Table/sum(Week_Pivot_Table)
        
        Percent_Week_Dict = dict(Percent_Week)
        TS_Week_Dict = dict(zip(Result.Week_Start,Result.BaseLine))
        
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
        Daily_Result = pd.DataFrame({'Date':Daily_Index[3:-3],'Actual Sales':New_TS.values,'Normal_Sales':Normal_Sales[3:-3]})
        
    else:
        TS_Week = pd.Series(Data.AMOUNT.values,index = Data.Week_Start)
        
        TS = pd.Series(Daily_Data.AMOUNT.values,index = Daily_Data.Date)
        
        New_TS = TS[datetime(2014,10,1):datetime(2014,12,31)]
        Week = [date.weekday() for date in New_TS.index]
        TS_DF = pd.DataFrame({'Date':New_TS.index,'AMOUNT':New_TS.values,'Week':Week})
        
        Week_Pivot_Table = TS_DF.pivot_table('AMOUNT','Week',aggfunc = 'mean')
        Percent_Week = Week_Pivot_Table/sum(Week_Pivot_Table)
        
        Percent_Week_Dict = dict(Percent_Week)
        TS_Week_Dict = dict(zip(Result.Week_Start,Result.BaseLine))
        
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
        
        Daily_Result = pd.DataFrame({'Date':Daily_Index[2:-4],'Actual Sales':New_TS.values,'Normal_Sales':Normal_Sales[2:-4]})    
    
    return Daily_Result
    
    
#--------2014------------------------------------------------------------------------------------------------    
    
Data = pd.read_excel('2014_NFS_Test.xlsx')
Daily_Data = pd.read_excel('2014_Apply_To_Daily.xlsx')

NFS_Daily_Split = pd.read_excel('NFS_Daily_Split.xlsx')
NFS_Week_Split = pd.read_excel('NFS_Week_Split.xlsx')

NFS_Col = NFS_Daily_Split.columns

for i in range(1,len(NFS_Col)):
    TS_Daily = pd.Series(NFS_Daily_Split.ix[:,i].values,index = NFS_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2014,10,1):datetime(2015,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NFS_Week_Split.ix[:,i].values,index = NFS_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2014,3,3):datetime(2015,3,23)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,2014)
#    Daily_Result.to_excel('.//NFS//2014_NFS_'+str(i)+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    
    Daily_Result.to_excel('.//NFS//2014_NFS_'+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    

#---------------------------------------------------------------------

Data = pd.read_excel('2014_NSO_Test.xlsx')
Daily_Data = pd.read_excel('2014_Apply_To_Daily.xlsx')

NSO_Daily_Split = pd.read_excel('NSO_Daily_City_Amount.xlsx')
NSO_Week_Split = pd.read_excel('NSO_ALL_Week_City_AMOUNT.xlsx')

NSO_Col = NSO_Daily_Split.columns

for i in range(1,len(NSO_Col)):
    TS_Daily = pd.Series(NSO_Daily_Split.ix[:,i].values,index = NSO_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2014,10,1):datetime(2015,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NSO_Week_Split.ix[:,i].values,index = NSO_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2014,3,3):datetime(2015,3,23)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,2014)
    Daily_Result.to_excel('.//NSO//2014_NSO_'+str(i)+NSO_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    


#-------2015----------------------------------------------------------------------------------------------------------------

Data = pd.read_excel('2015_NFS_Test.xlsx')
Daily_Data = pd.read_excel('2015_Apply_To_Daily.xlsx')

NFS_Daily_Split = pd.read_excel('NFS_Daily_Split.xlsx')
NFS_Week_Split = pd.read_excel('NFS_Week_Split.xlsx')

NFS_Col = NFS_Daily_Split.columns

for i in range(1,len(NFS_Col)):
    TS_Daily = pd.Series(NFS_Daily_Split.ix[:,i].values,index = NFS_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2015,10,1):datetime(2016,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NFS_Week_Split.ix[:,i].values,index = NFS_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2015,3,2):datetime(2016,3,21)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,2015)
#    Daily_Result.to_excel('.//NFS//2015_NFS_'+str(i)+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    
    Daily_Result.to_excel('.//NFS//2015_NFS_'+NFS_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)    

#---------------------------------------------------------------------

Data = pd.read_excel('2014_NSO_Test.xlsx')
Daily_Data = pd.read_excel('2014_Apply_To_Daily.xlsx')

NSO_Daily_Split = pd.read_excel('NSO_Daily_City_Amount.xlsx')
NSO_Week_Split = pd.read_excel('NSO_ALL_Week_City_AMOUNT.xlsx')

NSO_Col = NSO_Daily_Split.columns

for i in range(1,len(NSO_Col)):
    TS_Daily = pd.Series(NSO_Daily_Split.ix[:,i].values,index = NSO_Daily_Split.ix[:,0].values)
    TS_Daily_New = TS_Daily[datetime(2015,10,1):datetime(2016,3,31)]
    Daily_Data.AMOUNT = TS_Daily_New.values
    TS_Week = pd.Series(NSO_Week_Split.ix[:,i].values,index = NSO_Week_Split.ix[:,0].values)
    TS_Week_New = TS_Week[datetime(2015,3,2):datetime(2016,3,21)]
    Data.AMOUNT = TS_Week_New.values
    Daily_Result = Get_Base_Uplift(Data,Daily_Data,2015)
    Daily_Result.to_excel('.//NSO//2015_NSO_'+str(i)+NSO_Col[i]+'.xlsx',encoding='utf8',header=True,index = False)  