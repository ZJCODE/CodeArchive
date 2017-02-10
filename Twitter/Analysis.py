# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:01:06 2016

@author: ZJun
"""

from GetData import LoadData
from twitter_function import Import_Obj,GetPartOfTimeSeries,Sort_Dict_key
import pandas as pd
from GetFactors import GetTwitterInPlaceLoc,GetFluRelatedTwitterInPlaceLoc
from GetFactors import GetMoveDestinationInPlaceLoc,GetFluRelatedMoveDestinationInPlaceLoc,GetActuallDayInWeek
import matplotlib.pyplot as plt
from collections import Counter

week_move = Import_Obj('./Data/week_move')
week_user_flu_state = Import_Obj('./Data/week_user_flu_state')
Queensland_Flu = pd.read_csv('./Data/Queensland2015.csv')
from datetime import timedelta


def GetRealFlu(Queensland_Flu,location=None):
    first_week = pd.datetime(2015,1,5).date()
    t = [first_week]
    if location != None:        
        x = Queensland_Flu[location].values
    else:
        x = Queensland_Flu.ix[:,1:].sum(1).values
    for i in range(1,len(x)):
        t.append(first_week + timedelta(i*7))
    ts = pd.Series(x,index = t[:len(x)])
    return ts

# Raw

Data = LoadData()

a = Sort_Dict_key(Counter(Data.date))   
plt.plot([i[0] for i in a],[i[1] for i in a])

def main(loc):
    
    place = 'Queensland'
    location = loc # None
    
    '''
    hhs_loc = ['Cairns', 'Townsville', 'Mackay', 'Fitzroy', 'Wide Bay',
       'Sunshine Coast', 'Brisbane', 'Darling Downs', 'Moreton',
       'Gold Coast']
    '''
    
    actual_days_in_week = GetActuallDayInWeek(week_user_flu_state)
    flu_related_twitter_in_place_loc = GetFluRelatedTwitterInPlaceLoc(Data,place,location)
    twitter_in_place_loc = GetTwitterInPlaceLoc(Data,place,location)
    move_destination_in_place_loc = GetMoveDestinationInPlaceLoc(week_move,place,location)    
    flu_related_move_destination_in_place_loc = GetFluRelatedMoveDestinationInPlaceLoc(week_move,week_user_flu_state,place,location)
    real_flu = GetRealFlu(Queensland_Flu,location)
    
    
    TimeRange1 = [pd.datetime(2015,1,12).date(),pd.datetime(2015,3,16).date()]
    TimeRange2 = [pd.datetime(2015,1,12).date(),pd.datetime(2015,3,16).date()]
    
    # Process
    flu_related_twitter_in_place_loc_ = GetPartOfTimeSeries(flu_related_twitter_in_place_loc,TimeRange1)
    twitter_in_place_loc_ = GetPartOfTimeSeries(twitter_in_place_loc,TimeRange1)
    flu_related_move_destination_in_place_loc_ = GetPartOfTimeSeries(flu_related_move_destination_in_place_loc,TimeRange1)
    move_destination_in_place_loc_ = GetPartOfTimeSeries(move_destination_in_place_loc,TimeRange1)
    actual_days_in_week_ = GetPartOfTimeSeries(actual_days_in_week,TimeRange1)
    real_flu_ = GetPartOfTimeSeries(real_flu,TimeRange2)
    
    
    Compare = pd.DataFrame({'real_flu':real_flu_,
                         'flu_related_twitter_in_place_loc':(flu_related_twitter_in_place_loc_/actual_days_in_week_).values,
                         'twitter_in_place_loc':(twitter_in_place_loc_/actual_days_in_week_).values,
                         'flu_related_move_destination_in_place_loc':(flu_related_move_destination_in_place_loc_/actual_days_in_week_).values,
                         'move_destination_in_place_loc':(move_destination_in_place_loc_/actual_days_in_week_).values})
    return Compare
 
