# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:01:33 2016

@author: ZJun
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from twitter_function import MapLocation,AddJudgeFlu,GenerateDate
from collections import Counter

    

def GetTwitterInPlaceLoc(Data,place = None ,location = None):
    '''
    return timeseries
    '''
    if place != None:        
        Data = Data[Data.District == place]
    if location != None:
        Data['Location'] = [MapLocation(l,place) for l in Data.Location]
        Data = Data[Data.Location == location]

    ts = pd.Series(np.ones(len(Data)),index=Data.created_at.values)
    week_ts = ts.resample('W-SUN',how='sum')
    weeks = [d.date() - timedelta(6) for d in week_ts.index]
    ts_twitter_in_place_loc = pd.Series(week_ts.values,index= weeks)
    return ts_twitter_in_place_loc


def GetFluRelatedTwitterInPlaceLoc(Data,place=None,location=None):
    '''
    return timeseries
    '''
    if place != None:        
        Data = Data[Data.District == place]
    if location != None:
        Data['Location'] = [MapLocation(l,place) for l in Data.Location]
        Data = Data[Data.Location == location]
    
    Data = AddJudgeFlu(Data)

    print 'Flu Added'
    ts = pd.Series(Data.Judge.values,index=Data.created_at.values)
    week_ts = ts.resample('W-SUN',how='sum')
    weeks = [d.date() - timedelta(6) for d in week_ts.index]
    ts_flu_related_twitter_in_place_loc = pd.Series(week_ts.values,index= weeks)
    return ts_flu_related_twitter_in_place_loc



    
def GetMoveInPlace(Move,place):
    '''
    specific location's place
    '''
    if place == None:
        MoveInWhere = Move
    else:        
        MoveInWhere = [a for a in Move if (place in a[1] and place in a[2])]
    users = [a[0] for a in MoveInWhere]
    pairs = [(MapLocation(a[1].split(',')[0],place),MapLocation(a[2].split(',')[0],place)) for a in MoveInWhere]
    DfMove = pd.DataFrame({'user':users,'pairs':pairs})
    DfMove['same'] = [0 if l[0]==l[1] else 1 for l in DfMove.pairs]
    DfMove = DfMove[DfMove.same == 1][['pairs','user']]
    return DfMove
      
    

def GetMoveDestinationInPlaceLoc(week_move,place,location = None):
    '''
    return timeseries
    '''   
    weeks = sorted(week_move.keys())
    count = []
    for w in weeks:
        move = week_move[w]
        move_in_place = GetMoveInPlace(move,place)
        destination = [i[1] for i in move_in_place.pairs]
        if location != None:            
            count.append(destination.count(location))
        else:
            count.append(len(destination))
    ts = pd.Series(count,index = weeks)
    return ts


def GetWhere2WhereMatrix(count_where_to_where):

    hhs_loc = ['Cairns', 'Townsville', 'Mackay', 'Fitzroy', 'Wide Bay',
   'Sunshine Coast', 'Brisbane', 'Darling Downs', 'Moreton',
   'Gold Coast']
    where_to_where_matrix = np.zeros([len(hhs_loc),len(hhs_loc)])  
    for i,loc1 in enumerate(hhs_loc):
        for j,loc2 in enumerate(hhs_loc):
            where_to_where_matrix[i,j] += count_where_to_where[(loc1,loc2)]
    df_where_to_where = pd.DataFrame(where_to_where_matrix,index = hhs_loc,columns = hhs_loc)
    return df_where_to_where
        
def GetMoveInPlaceWhere2Where(week_move,place):
    '''
    return a series of dataframe
    ''' 
    weeks = sorted(week_move.keys())
    where_to_where = []
    for w in weeks:
        move = week_move[w]
        move_in_place = GetMoveInPlace(move,place)
        count_where_to_where = Counter(move_in_place.pairs)
        where_to_where.append(GetWhere2WhereMatrix(count_where_to_where))

    ts = pd.Series(where_to_where,index = weeks)
    return ts
    

def GetFluRelatedMoveDestinationInPlaceLoc(week_move,week_user_flu_state,place,location=None):
    '''
    return timeseries
    '''
    weeks = sorted(week_move.keys())
    count = []
    for w in weeks:
        move = week_move[w]
        move_in_place = GetMoveInPlace(move,place)
        flu_state = week_user_flu_state[w]
        flu_users = flu_state[0]
        Judge = [1 if user in flu_users else 0 for user in move_in_place.user]
        move_in_place['Judge'] = Judge
        flu_related_data = move_in_place[move_in_place.Judge==1]
        destination = [i[1] for i in flu_related_data.pairs]
        if location != None:            
            count.append(destination.count(location))
        else:
            count.append(len(destination))
    ts = pd.Series(count,index = weeks)
    return ts


def GetFluRelatedMoveInPlaceWhere2Where(week_move,week_user_flu_state,place):
    '''
    return a series of dataframe
    '''  
    weeks = sorted(week_move.keys())
    where_to_where = []
    for w in weeks:
        move = week_move[w]
        move_in_place = GetMoveInPlace(move,place)
        flu_state = week_user_flu_state[w]
        flu_users = flu_state[0]
        Judge = [1 if user in flu_users else 0 for user in move_in_place.user]
        move_in_place['Judge'] = Judge
        flu_related_data = move_in_place[move_in_place.Judge==1]
        count_where_to_where = Counter(flu_related_data.pairs)
        where_to_where.append(GetWhere2WhereMatrix(count_where_to_where))

    ts = pd.Series(where_to_where,index = weeks)
    return ts


def GetActuallDayInWeek(week_user_flu_state):
    weeks = sorted(week_user_flu_state.keys())
    actual_day = []
    for w in weeks:
        actual_day.append(week_user_flu_state[w][1])
    ts = pd.Series(actual_day,index=weeks)
    return ts


   