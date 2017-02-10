# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:38:43 2016

@author: ZJun
"""

from GetData import LoadData
from twitter_function import Import_Obj,GetPartOfTimeSeries,Sort_Dict_key,Save_Obj
import pandas as pd
import numpy as np
from GetFactors import GetTwitterInPlaceLoc,GetFluRelatedTwitterInPlaceLoc,GetMoveInPlaceWhere2Where,GetFluRelatedMoveInPlaceWhere2Where
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


actual_days_in_week = GetActuallDayInWeek(week_user_flu_state)
    
place = 'Queensland'

hhs_loc = ['Cairns', 'Townsville', 'Mackay', 'Fitzroy', 'Wide Bay',
   'Sunshine Coast', 'Brisbane', 'Darling Downs', 'Moreton',
   'Gold Coast']
    
TimeRange = [pd.datetime(2015,1,12).date(),pd.datetime(2015,7,20).date()]



num_week = ((TimeRange[1] - TimeRange[0])/ 7).days + 1

flu_related_twitter_in_place_loc = np.ones(num_week)
twitter_in_place_loc = np.ones(num_week)
move_destination_in_place_loc = np.ones(num_week)
flu_related_move_destination_in_place_loc = np.ones(num_week)
real_flu = np.ones(num_week)

for location in hhs_loc:

    flu_related_twitter_in_place_loc_ = GetPartOfTimeSeries(GetFluRelatedTwitterInPlaceLoc(Data,place,location),TimeRange)
    twitter_in_place_loc_ = GetPartOfTimeSeries(GetTwitterInPlaceLoc(Data,place,location),TimeRange)
    move_destination_in_place_loc_ = GetPartOfTimeSeries(GetMoveDestinationInPlaceLoc(week_move,place,location),TimeRange)    
    flu_related_move_destination_in_place_loc_ = GetPartOfTimeSeries(GetFluRelatedMoveDestinationInPlaceLoc(week_move,week_user_flu_state,place,location),TimeRange)
    real_flu_ = GetPartOfTimeSeries(GetRealFlu(Queensland_Flu,location),TimeRange)
    time_index = real_flu_.index
    flu_related_twitter_in_place_loc = np.c_[flu_related_twitter_in_place_loc,flu_related_twitter_in_place_loc_]
    twitter_in_place_loc = np.c_[twitter_in_place_loc,twitter_in_place_loc_]
    move_destination_in_place_loc = np.c_[move_destination_in_place_loc,move_destination_in_place_loc_]
    flu_related_move_destination_in_place_loc = np.c_[flu_related_move_destination_in_place_loc,flu_related_move_destination_in_place_loc_]
    real_flu = np.c_[real_flu,real_flu_]
    

df_flu_related_twitter_in_place_loc = pd.DataFrame(flu_related_twitter_in_place_loc[:,1:],columns=hhs_loc,index = time_index)
df_twitter_in_place_loc = pd.DataFrame(twitter_in_place_loc[:,1:],columns=hhs_loc,index = time_index)
df_move_destination_in_place_loc = pd.DataFrame(move_destination_in_place_loc[:,1:],columns=hhs_loc,index = time_index)
df_flu_related_move_destination_in_place_loc = pd.DataFrame(flu_related_move_destination_in_place_loc[:,1:],columns=hhs_loc,index = time_index)
df_real_flu = pd.DataFrame(real_flu[:,1:],columns=hhs_loc,index = time_index)

df_where_to_where = GetMoveInPlaceWhere2Where(week_move,place)
df_flu_where_to_where = GetFluRelatedMoveInPlaceWhere2Where(week_move,week_user_flu_state,place)
    
    

Save_Obj(df_flu_related_twitter_in_place_loc,'./DF_Result/df_flu_related_twitter_in_place_loc')
Save_Obj(df_twitter_in_place_loc,'./DF_Result/df_twitter_in_place_loc')
Save_Obj(df_move_destination_in_place_loc,'./DF_Result/df_move_destination_in_place_loc')
Save_Obj(df_flu_related_move_destination_in_place_loc,'./DF_Result/df_flu_related_move_destination_in_place_loc')
Save_Obj(df_real_flu,'./DF_Result/df_real_flu')
Save_Obj(df_where_to_where,'./DF_Result/df_where_to_where')
Save_Obj(df_flu_where_to_where,'./DF_Result/df_flu_where_to_where')
    


 
