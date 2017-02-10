# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:25:22 2016

@author: ZJun
"""
import psycopg2 as db
import pandas as pd
import numpy as np
from twitter_function import Save_Obj,AddUserName,AddDate,AddJudgeFlu

def GetDataText():
    
    conn = db.connect("host=localhost dbname=aussie_twitter user=postgres password=dbzjun client_encoding='utf-8'")
    # conn.set_character_set('utf8')
    cur = conn.cursor()         
    SELECT_QUERY="""
    SELECT uid,text,created_at
    FROM aussie_tweets 
    ORDER BY created_at
    """
    try:
        cur.execute(SELECT_QUERY)
        rows = cur.fetchall()
        X = np.array(rows)
    except db.Error, e:
        print "Error ocurred: %s " % e.args[0]
        print e     
    data_text = pd.DataFrame(X,columns=['uid','text','created_at'])
    return data_text

def UserFluStateByTime(data_text,way = 'week'):    
    AddDate(data_text)
    AddUserName(data_text)
    AddJudgeFlu(data_text)
    #data_text['Judge'] = np.ones(len(data_text))
    from datetime import timedelta
    data_text = data_text.sort_values(['date','user_name'])
    if way == 'week':
        days = sorted(list(set(data_text.date))) 
        min_day =  min(days)
        if min_day.weekday() == 0:
            date_start = min_day
        else:            
            date_start = min_day + timedelta(7-min_day.weekday())
        day_7 = timedelta(7)
        date_end = date_start + day_7
        t = [date_start]
        flu_user = []
        actual_day = [] # how many days in that week
        for i in range((max(days) - min(days)).days/7):
            print str(i+1) + ' Week Finished'
            data = data_text[(data_text.date >= date_start)&(data_text.date < date_end)]
            actual_day.append(len(set(data.date)))
            flu_data = data[data.Judge == 1]
            flu_user.append(list(set(flu_data.user_name)))  # Counter 
            t.append(date_start)
            date_start = date_end
            date_end = date_start + day_7
        flu_user_and_days = zip(flu_user,actual_day)  # some day missing
        return dict(zip(t,flu_user_and_days))
        
    elif way == 'day':
        days = sorted(list(set(data_text.date)))
        flu_user = []
        for day in days:
            data = data_text[data_text.date == day]
            flu_data = data[data.Judge == 1]
            flu_user.append(list(set(flu_data.user_name)))
            print str(day) + ' Finished '
        return dict(zip(days,flu_user))
            
    elif way == 'month':
        t = range(1,9)
        flu_user = []
        actual_day = [] # how many days in that month
        data_text['month'] = [date.month for date in data_text.date]
        for month in t:
            data = data_text[data_text.month == month]
            actual_day.append(len(set(data.date)))
            flu_data = data[data.Judge == 1]
            flu_user.append(list(set(flu_data.user_name)))
            print str(month) + ' Month Finished'
        flu_user_and_days = zip(flu_user,actual_day)
        return dict(zip(t,flu_user_and_days))    
    else:
        print 'Input Error Way'
   

data_text = GetDataText()
week_user_flu_state = UserFluStateByTime(data_text,'week')
Save_Obj(week_user_flu_state,'week_user_flu_state')
     
        