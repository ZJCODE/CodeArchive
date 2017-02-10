# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:40:57 2016

@author: ZJun
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#Index([u'passenger_ID2', u'flight_ID', u'flight_time', u'checkin_time',u'flight_Type'])
Departure = pd.read_csv('./Data/airport_gz_departure_chusai_1stround_processed.csv',parse_dates=['flight_time','checkin_time'])

def SplitDataByEach10Min(Data,timecolumn):
    Data = Data.dropna(subset = [timecolumn]) # Drop Nan Data
    Data = Data.sort_values(by = timecolumn)
    Data.index = range(len(Data))
    TimeSetList = list(set(Data[timecolumn]))
    MinTime = min(TimeSetList) 
    MaxTime = max(TimeSetList)

    def Ceil_Time(time):
        '''
        set 2016-9-12 12:43:10 to 2016-9-12 12:40:00 like this
        '''
        from datetime import timedelta

        def Celi_Minute(Minute):
            '''
            Set 45 to 40 \ 34 to 30 like this
            '''
            return int(10*(round(Minute/10)+1))

        try:            
            T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)+ timedelta(0,0,0,0,Celi_Minute(time.minute),0)
        except:
            try:            
                T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)
            except:
                T = time
        return T
    
    #Initial Time Range with Freq = 10Min
    TimeRange = pd.date_range(start = Ceil_Time(MinTime) , end = Ceil_Time(MaxTime) , freq = '10Min') # Freq : 10Min , 5H , ....
        
    TimeSplitData = [] # Used for storing split data according to time slice
    
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
    

def GetDepartureCountData(TimeColumn):
    '''
    # Input 'flight_time' or 'checkin_time' 
    '''
    TimeSplitData , TimeRange = SplitDataByEach10Min(Departure,TimeColumn)
     
          
    FlightTime_Count = TimeSplitData[0].pivot_table(values = 'flight_ID' , index = 'flight_Type' , aggfunc = 'count')
    SplitTime = [TimeRange[0]]*len(FlightTime_Count)
    
    DF_FlightTime_Count = pd.DataFrame({'flight_Type':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
    
    FlightTime_Count_All = DF_FlightTime_Count
    
    for i in range(1,len(TimeRange)-1):
        FlightTime_Count = TimeSplitData[i].pivot_table(values = 'flight_ID' , index = 'flight_Type' , aggfunc = 'count')
        if len(FlightTime_Count) > 0:
            SplitTime = [TimeRange[i]]*len(FlightTime_Count)   
            DF_FlightTime_Count = pd.DataFrame({'flight_Type':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
        else:
            SplitTime = [TimeRange[i]]
            DF_FlightTime_Count = pd.DataFrame({'flight_Type':[np.nan],'FlightPassengerCount':[0],'Time':SplitTime})
        
        FlightTime_Count_All = pd.concat([FlightTime_Count_All,DF_FlightTime_Count])
    
    return FlightTime_Count_All ,TimeRange
    
    
def Generate_flight_checkin_count():
    
    flight_time_count , flight_time_range = GetDepartureCountData('flight_time')
    checkin_time_count , checkin_time_range = GetDepartureCountData('checkin_time')
    
    path1 = './Data/flight_time_count.csv'
    path2 = './Data/checkin_time_count.csv'
    
    def Save_DataFrame_csv(DF,File_Path):
        DF.to_csv(File_Path,encoding='utf8',header=True,index = False)
    
    Save_DataFrame_csv(flight_time_count,path1)
    Save_DataFrame_csv(checkin_time_count,path2)


    '''
    Output
    FlightPassengerCount,Time,flight_Type
    206,2016-09-11 00:30:00,CZ
    This means at 2016-9-11 00:30:00 flight company CZ has some planes to departure ,total 206 people
    '''


if __name__ == '__main__':
    Generate_flight_checkin_count()

#flight_time_count[flight_time_count.flight_Type=='CZ'].plot(x='Time',y = 'FlightPassengerCount')
#checkin_time_count[checkin_time_count.flight_Type=='CZ'].plot(x='Time',y = 'FlightPassengerCount')



'''

#flight_time_count.plot(x='Time',y='FlightCount')
#checkin_time_count.plot(x='Time',y='FlightCount')

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L

Flight_Type_list = [a[0] for a in Sort_Dict(Counter(flight_time_count.flight_Type))]
t = pd.Series([0]*len(flight_time_range),index=flight_time_range)
test = flight_time_count.pivot_table('FlightPassengerCount',index = ['flight_Type','Time'],aggfunc=sum)

a = test[Flight_Type_list[6]]
m =a+t
m = m.fillna(0)
m.plot()

'''