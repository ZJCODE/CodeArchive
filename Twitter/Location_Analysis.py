# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:10:28 2016

@author: ZJun
"""

import pandas as pd
from GetData import getData
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
from datetime import timedelta
from Net import GenerateNetwork,DrawGraph,GetCoreSubNetwork,CommunityDetection

def Import_Obj(File):    
    import pickle
    File_Name = File+'.pkl'
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)
    

def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()    

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L
    
def AddUserName(Data):
    id_name_dict = Import_Obj('./Data/IdNameDict')
    user_name = [id_name_dict[Id] for Id in Data.uid]
    Data['user_name'] = user_name
    
def AddDate(Data):
    d = [d.date() for d in Data.created_at]
    Data['date'] = d
    
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
    return move_position_list,move_time_list

# p,t = GetMovePositionList(GetUserData(Data,Data.user_name[23459])) 
 
def MoveDirectionWithTime(move_position_list,move_time_list):
    '''
    where to where and when
    '''
    return [[(move_position_list[i],move_position_list[i+1]),move_time_list[i+1].date()] for i in range(len(move_position_list)-1)]

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
    return move_list
            
            

        
def DrawTrack(u_data):
    position_list = np.c_[u_data.latitude.values,u_data.longitude.values]
    plt.plot(position_list[:,0],position_list[:,1],'g-',alpha=0.7)
    
    
    
def GenerateDate(year,month,day):
    return pd.datetime(year,month,day).date()
    
def GetDateData(Data,date):
    return Data[Data.date == date]
  

def DeleteMoveListException(move_list):
    
    '''
    remove some exception
    like None 
    or some pairs with different description but actually the same place in where to where
    '''
    
    df_move_list = pd.DataFrame(move_list,columns = ['pair','time'])
    flag = []
    for pair in df_move_list.pair:
        if pair[0] == None or pair[1] == None:
            flag.append(np.nan)
        else:            
            try:            
                p_1_1 , p_1_2 = [a.strip() for a in pair[0].split(',')]
                p_2_1 , p_2_2 = [a.strip() for a in pair[1].split(',')]
                if p_1_1 == p_2_2 or p_1_2 == p_2_1:
                    flag.append(np.nan)
                else:
                    flag.append(1)
            except:
                flag.append(1)
    df_move_list['flag'] = flag
    df_move_list_without_exception = df_move_list.dropna()[['pair','time']]
    df_move_list_without_exception.index = range(len(df_move_list_without_exception))
    return df_move_list_without_exception

def AggMoveListByTime(move_list,way = 'week'):
    
    '''
    aggregate move_list by a given time type
    return a dictionary which can be use to 
    retrive move pairs in a specific day or week or month
    '''
    
    if way == 'week':
        days = sorted(list(set(move_list.time))) 
        date_start = min(days)
        day_7 = timedelta(7)
        date_end = date_start + day_7
        t = [date_start + timedelta(1)]
        pairs = []
        for i in range((max(days) - min(days)).days/7):
            print str(i+1) + ' Week Finished'
            pairs.append(list(move_list[(move_list.time > date_start)&(move_list.time <= date_end)].pair.values))
            t.append(date_start + timedelta(1))
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
    Data = getData()
    AddDate(Data)
    AddUserName(Data)
    move_list = DeleteMoveListException(GetAllMove_FasterVersion(Data))
    day_m = AggMoveListByTime(move_list,way='day')
    week_m = AggMoveListByTime(move_list,way='week')
    month_m = AggMoveListByTime(move_list,way='month')
    Save_Obj(day_m,'day_move')
    Save_Obj(week_m,'week_move')
    Save_Obj(month_m,'month_move')
    
    
    
    


