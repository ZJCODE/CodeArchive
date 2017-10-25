# -*- coding: utf-8 -*-
"""
@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn
import time
import datetime


def deal_with_big_error(ts,up,down,draw = False):
    ts = ts.fillna(method = 'backfill').fillna(method='ffill')
    ts_old = ts.copy()
    ts_up = ts.quantile(up)   
    ts_down = ts.quantile(down)
    ts[ts>ts_up] = np.nan
    ts = ts.fillna(ts.max())
    ts[ts<ts_down] = np.nan
    ts = ts.fillna(ts.min())
    if draw == True:        
        plt.plot(ts_old,alpha = 0.7)
        plt.plot(ts,alpha=0.5)
    return ts

def quadratic_fun_fit(ts):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X=np.zeros([len(ts),2])
    X[:,0] = range(len(ts))
    X[:,1] = [i*i for i in range(len(ts))]
    y = ts.values
    model = LinearRegression()
    model.fit(X,y)
    y_hat = model.predict(X)
    return model,y_hat

def deal_error_enhanced(ts,up,down):
    model,y_hat = quadratic_fun_fit(ts)
    no_trend_ts = ts / y_hat
    no_trend_ts_remove_big_error = deal_with_big_error(no_trend_ts,up,down,draw=False)
    new_ts = y_hat * no_trend_ts_remove_big_error
    return new_ts   

def linear_fun_fit(ts):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = np.arange(len(ts))[:,np.newaxis]
    y = ts.values
    model = LinearRegression()
    model.fit(X,y)
    y_hat = model.predict(X)
    y_hat_ts = pd.Series(y_hat.reshape(1,-1)[0],index = ts.index)
    plt.plot(ts)
    plt.plot(y_hat_ts)
    return model,y_hat


def predict_max(data,feature,look_back_day,previous_start_time,previous_end_time,predict_start_time,predict_end_time,filter_up,filter_down,last_time_promotional_day = False,today=False):
    '''
    Input:
    data 数据源
    feature 具体预测的维度，比如呼叫量、应答量、市内呼叫、跨城呼叫
    look_back_day 训练模型使用的数据量，今日往前多少天
    previous_start_time 去年节日起始时间
    previous_end_time 去年节日结束时间
    predict_start_time 本次节日起始时间
    predict_end_time 本次节日结束时间
    filter_up 过滤异常的阈值上限
    filter_down 过滤异常的阈值下限
    last_time_promotional_day 上一次做活动时间
    '''
    plt.rc('figure',figsize=[20,10])
    plt.subplot(211)
    data_previous_time = data[feature][pd.to_datetime(previous_start_time) - datetime.timedelta(days = 15):pd.to_datetime(previous_end_time) + datetime.timedelta(days = 15)]
    plt.plot(data_previous_time)
    for t in [previous_start_time,previous_end_time]:
        plt.plot([t,t],[min(data_previous_time),max(data_previous_time)],'r',alpha  = 0.2)    
    plt.title('Last Year\'s Situation for Reference',fontsize = 16) 


    
    
    if today == False:
        today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    data_for_model = data[feature][pd.to_datetime(today) - datetime.timedelta(days = look_back_day):today]
    
    plt.subplot(212)
    
    model,_ = linear_fun_fit(deal_error_enhanced(data_for_model,filter_up,filter_down))
    t_s = str((pd.to_datetime(today) - datetime.timedelta(days = look_back_day)).date())
    t_e = str((pd.to_datetime(predict_start_time) - datetime.timedelta(days = 1)).date())
    predict_time_index = pd.date_range(t_s,t_e)
    X_p = np.arange(len(predict_time_index))[:,np.newaxis]
    ts_p = pd.Series(model.predict(X_p).reshape(1,-1)[0],index=predict_time_index)
    
    df_ts=pd.DataFrame()
    data_for_week = data_for_model[data_for_model / deal_error_enhanced(data_for_model,filter_up,filter_down) == 1]
    df_ts['value'] = data_for_week
    df_ts['week'] = [d.weekday() for d in data_for_week.index]
    week_period = df_ts.pivot_table('value','week',aggfunc='mean')
    #week_period.plot()
    week_ratio = week_period/week_period.mean()
    
    
    ts_pd = pd.DataFrame()
    ts_pd['value'] = ts_p
    ts_pd['week'] = [d.weekday() for d in ts_p.index]
    ts_pd['new_value'] = [v*week_ratio.loc[w,:].values[0] for v,w in zip(ts_pd.value,ts_pd.week)]
    #ts_pd['new_value'] = [v*week_ratio[w] for v,w in zip(ts_pd.value,ts_pd.week)]
    ts_pd_p = ts_pd.new_value
    
    
    if last_time_promotional_day == False:
        r1 = 1
    else:   
        last_time_promotional_day_s = (pd.to_datetime(last_time_promotional_day) - datetime.timedelta(days = 1)).date()
        last_time_promotional_day_e = (pd.to_datetime(last_time_promotional_day) + datetime.timedelta(days = 7)).date()
        last_time_promotional_day_s_week_ago = last_time_promotional_day_s - datetime.timedelta(days = 8)
        last_time_promotional_day_e_week_ago = last_time_promotional_day_s - datetime.timedelta(days = 1)

        last_time_promotional_day_compare_s = last_time_promotional_day_s - datetime.timedelta(days = 365)
        last_time_promotional_day_compare_e = last_time_promotional_day_e - datetime.timedelta(days = 365)
        last_time_promotional_day_compare_s_week_ago = last_time_promotional_day_s_week_ago - datetime.timedelta(days = 365)
        last_time_promotional_day_compare_e_week_ago = last_time_promotional_day_e_week_ago - datetime.timedelta(days = 365)
        
        a1 = max(data[feature][last_time_promotional_day_s:last_time_promotional_day_e])*1.0 / max(data[feature][last_time_promotional_day_s_week_ago:last_time_promotional_day_e_week_ago])
        a2 = max(data[feature][last_time_promotional_day_compare_s:last_time_promotional_day_compare_e])*1.0 / max(data[feature][last_time_promotional_day_compare_s_week_ago:last_time_promotional_day_compare_e_week_ago])
        r1 = a1/a2

    # 去年同个节假日

    previous_start_time_around = (pd.to_datetime(previous_start_time) - datetime.timedelta(days = 2)).date()
    previous_end_time_around = (pd.to_datetime(previous_end_time) + datetime.timedelta(days = 2)).date()
    previous_start_time_week_ago = previous_start_time_around - datetime.timedelta(days = 15)
    previous_end_time_week_ago = previous_start_time_around - datetime.timedelta(days = 1)

    r2 = max(data[feature][previous_start_time_around:previous_end_time_around])*1.0 / max(data[feature][previous_start_time_week_ago:previous_end_time_week_ago])
    if feature == 'cross_city_call':
        max_p = max((ts_pd_p[-7:])*r1*r2)*0.84
    elif feature == 'city_call':
        max_p = max((ts_pd_p[-7:])*r1*r2)
    else:
        max_p = max((ts_pd_p[-7:])*r1*r2)

    plt.plot(ts_pd_p,alpha = 0.5)
    plt.plot(data_for_model,'p')
    plt.legend(['filter ts','trend','trend_week','original data'],fontsize = 15,loc='upper left')
    plt.title('Max Value During Hoilday is  : %d , promotional_ratio : %.3f , hoilday_ratio : %.3f '%(max_p,r1,r2),fontsize = 16) 
    plt.show()

    return max_p



if __name__ == '__main__':
    import os
    data = pd.read_excel('data.xlsx')
    data.index = data.time
    while True:
        os.system('clear')
        Q = raw_input('\nInput quit() to quit else tap Enter :  ' )
        if Q == 'quit()':
            os._exit(1) 
        else:
            feature = raw_input('feature (call,city_call,cross_city_call,answer): ')
            look_back_day = input('look_back_day : ')

            previous_start_time = raw_input('previous_start_time (yyyy-mm-dd) : ')
            previous_end_time = raw_input('previous_end_time (yyyy-mm-dd) : ')

            predict_start_time = raw_input('predict_start_time (yyyy-mm-dd) : ')
            predict_end_time = raw_input('predict_end_time (yyyy-mm-dd) : ')
            filter_up = input('filter_up : ')
            filter_down = input('filter_down : ')
            last_time_promotional_day = raw_input('has promotional activity input last_time_promotional_day (yyyy-mm-dd) else input no : ')
            #today = '2017-4-15'
            today = raw_input('Now : input now or someday (yyyy-mm-dd) : ')

            if today == 'now' or today == 'Now':
                today = False
            if last_time_promotional_day == 'no':
                last_time_promotional_day = False

            max_p = predict_max(data,feature,look_back_day,previous_start_time,previous_end_time,predict_start_time,predict_end_time,filter_up,filter_down,last_time_promotional_day,today)
