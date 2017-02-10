# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:10:28 2016

@author: ZJun
"""

import pandas as pd
from GetData import GetData,AddPos
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
from collections import Counter
from twitter_function import Import_Obj,Save_Obj,Sort_Dict,AddUserName,AddDate,GenerateDate,Sort_Dict_key

    
   
    
def GetUserData(Data,user_name):
    return Data[Data.user_name == user_name]
    
def GetUserMovePositionList(u_data):
    '''
    input data about a specific user
    output this user's move situation
    '''
    location = u_data.location_name.values
    time = u_data.created_at.values
    move_position_list = [location[0]]
    move_time_list = [time[0]]
    j=0
    for i in range(1,len(u_data)):
        if location[i] == move_position_list[j]:
            i=i+1
            pass
        else:
            move_position_list.append(location[i])
            move_time_list.append(time[i])
            i=i+1
            j=j+1
    u_date_names = list(set(u_data.user_name))*len(move_position_list)
    return move_position_list,move_time_list,u_date_names

# p,t = GetMovePositionList(GetUserData(Data,Data.user_name[23459])) 
 
def MoveDirectionWithTime(move_position_list,move_time_list,u_date_names):
    '''
    where to where and when
    '''
    return [[(u_date_names[i],move_position_list[i],move_position_list[i+1]),move_time_list[i+1].date()] for i in range(len(move_position_list)-1)]

# MoveDirectionWithTime(p,t)
# MoveDirectionWithTime(*GetMovePositionList(GetUserData(Data,Data.user_name[23459])))


def GetAllMove(Data):    
    all_users = list(set(Data.user_name))
    move_list=[]
    i=0
    for user in all_users:
        t1 = time.time()
        move_list += MoveDirectionWithTime(*GetUserMovePositionList(GetUserData(Data,user)))
        i=i+1
        t2 = time.time()
        print '======user ' +str(i) + ' is finished '+ 'cost ' + str(t2-t1) + ' seconds====='
    return move_list

   
def GetAllMove_FasterVersion(Data):
    '''
    Get All Move Pairs With TimeStamp (Day)
    '''
    t1 = time.time()
    sort_data =  Data.sort_values(by=['user_name','created_at'])
    t2 = time.time()
    print '=====Data Sort Finished====='+'Cost '+str(t2-t1) + ' Seconds ====='
    sort_data.index = range(len(sort_data))
    user_list = sort_data.user_name.values
    move_list=[]
    users = [user_list[0]]
    start = 0
    end = 0
    users_index = 0
    for i in range(len(user_list)):
        if users[users_index] == user_list[i]:
            pass
        else:
            t1 = time.time()
            users.append(user_list[i])
            users_index += 1
            end = i-1
            # Get Specific User's data [time series data with position]
            u_data = sort_data.ix[start:end,:]  
            move_list += MoveDirectionWithTime(*GetUserMovePositionList(u_data))
            start = i
            t2 = time.time()
            print '======user ' + str(users_index) + ' is finished '+ 'cost ' + str(t2-t1) + ' seconds====='
    move_list.sort(key= lambda x : x[1])     
    df_move_list = pd.DataFrame(move_list,columns = ['pair','time'])
    return df_move_list

        
def DrawTrack(u_data):
    position_list = np.c_[u_data.latitude.values,u_data.longitude.values]
    plt.plot(position_list[:,0],position_list[:,1],'g-',alpha=0.7)
        
    
def GetDateData(Data,date):
    return Data[Data.date == date]


def AggMoveListByTime(move_list,way = 'week'):
    
    '''
    aggregate move_list by a given time type
    return a dictionary which can be use to 
    retrive move pairs in a specific day or week or month
    '''
    
    if way == 'week':
        days = sorted(list(set(move_list.time))) 
        min_day =  min(days)
        if min_day.weekday() == 0:
            date_start = min_day
        else:            
            date_start = min_day + timedelta(7-min_day.weekday())
        day_7 = timedelta(7)
        date_end = date_start + day_7
        t = [date_start]
        pairs = []
        for i in range((max(days) - min(days)).days/7):
            print str(i+1) + ' Week Finished'
            pairs.append(list(move_list[(move_list.time >= date_start)&(move_list.time < date_end)].pair.values))
            t.append(date_start)
            date_start = date_end
            date_end = date_start + day_7   
            
    elif way == 'day':
        t = sorted(list(set(move_list.time)))   # Some days missed
        pairs = []
        for day in t:
            pairs.append(list(move_list[move_list.time == day].pair.values))
            print str(day) + ' Finished '
            
    elif way == 'month':
        t = range(1,9)
        pairs = []
        move_list['month'] = [date.month for date in move_list.time]
        for month in t:
            print str(month) + ' Month Finished'
            pairs.append(list(move_list[move_list.month == month].pair.values))
    
    else:
        print 'Input Error Way'
        
    agg_move_list_time_dict = dict(zip(t,pairs))
    
    return agg_move_list_time_dict



    
def main():
    Data = GetData()
    Data = AddPos(Data)
    AddDate(Data)
    
    # The amount of twitters
    a = Sort_Dict(Counter(Data.date))   
    plt.plot([i[0] for i in a],[i[1] for i in a])
    
    AddUserName(Data) 
    move_list = GetAllMove_FasterVersion(Data)
    day_m = AggMoveListByTime(move_list,way='day')
    week_m = AggMoveListByTime(move_list,way='week')
    month_m = AggMoveListByTime(move_list,way='month')
    Save_Obj(day_m,'./Data/day_move')
    Save_Obj(week_m,'./Data/week_move')
    Save_Obj(month_m,'./Data/month_move')


    
'''  
if __name__ == '__main__':
    main() 
'''



'''
def test_latitude_longitude():
    Data = getData()
    plt.scatter(Data.latitude[:100000],Data.longitude[:100000],alpha=0.01)
    AddDate(Data)
    AddUserName(Data)    
    loc = [(la,lo) for la,lo in zip(Data.latitude,Data.longitude)]
    Data['location_name'] =loc
    move_list = GetAllMove_FasterVersion(Data)   
    move_list = pd.DataFrame(move_list,columns = ['pair','time'])
    week_m = AggMoveListByTime(move_list,way='week')
    have_a_look = week_m[pd.datetime(2015,1,27).date()]
    for road in have_a_look[:1000]:
        plt.plot((road[0][0],road[1][0]),(road[1][1],road[0][1]))
'''
        

