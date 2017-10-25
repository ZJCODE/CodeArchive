#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:04:26 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import time
import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
LinearModel = LinearRegression()
import matplotlib.pyplot as plt
import time
import datetime

def linear_fun_fit(ts,draw = False):
    '''
    对时间序列加一个线性估计
    返回模型和预估值，其中模型可用于对未来趋势的预测
    '''
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = np.arange(len(ts))[:,np.newaxis]
    y = ts.values
    model = LinearRegression()
    model.fit(X,y)
    y_hat = model.predict(X)
    y_hat_ts = pd.Series(y_hat.reshape(1,-1)[0],index = ts.index)
    if draw == True:        
        plt.plot(ts)
        plt.plot(y_hat_ts)
    return model,y_hat

def quadratic_fun_fit(ts):
    '''
    对时间序列加一个二次函数估计
    返回模型和预估值，其中模型可用于对未来趋势的预测
    '''
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

def deal_with_big_error(ts,up,down,draw = False):
    '''
    对平稳序列基于百分位数去除异常值
    '''
    ts = ts.fillna(method = 'backfill').fillna(method='ffill')
    ts_old = ts.copy()
    ts_up = ts.quantile(up)   
    ts_down = ts.quantile(down)
    ts[ts>ts_up] = np.nan
    ts = ts.fillna(ts.max())
    ts[ts<ts_down] = np.nan
    ts = ts.fillna(ts.min())
    if draw == True:        
        plt.plot(ts_old,alpha = 0.3)
        plt.plot(ts)
    return ts

        
def fillna_ts(ts):
    '''
    填充空值
    '''
    ts = ts.fillna(method = 'backfill').fillna(method='ffill')
    return ts


def get_part_of_ts(ts,start_day = 0,end_day = -1):
    ts = ts[start_day:end_day]
    return ts


def ts_filter(ts,up=0.95,down=0.05):
    '''
    将带趋势序列的波动部分抽离去除异常后合并回时间序列
    '''
    ts = ts.fillna(method = 'backfill').fillna(method='ffill')
    model,y_hat = quadratic_fun_fit(ts)
    no_trend_ts = ts / y_hat
    no_trend_ts_remove_big_error = deal_with_big_error(no_trend_ts,up,down,draw=False)
    new_ts = y_hat * no_trend_ts_remove_big_error
    return new_ts


def ts_plot(ts,start_day = 0,end_day = -1,show_week_line = False):
    ts = ts[start_day:end_day]
    up = max(ts)
    down = min(ts)
    plt.rc('figure',figsize = [16,10])
    if show_week_line == True:            
        Index = pd.date_range(start=ts.index[0],end = ts.index[-1],freq='W-MON')
        for x in Index:
            plt.plot([x,x],[down,up],'-.k',alpha=0.2)   
    plt.plot(ts)


def day_enconding(ts,kind=False,day_holiday=False,day_spring_festival=False,day_promotion=False):
    '''
    
    Desc:
    
    encoding what kind of day it is
    
    InPut:
    
    [ts] time series
    [kind]  include 'week'  , 'holiday' , 'spring_festival' , 'promotion'
    [day_holiday] holiday day list 
    [day_spring_festival] spring festival day list
    [day_promotion] promotional activities day list 
    
    Output:

    day_enconding_feature
    
    '''
    if kind == 'week':
        day_enconding_feature = np.array([d.weekday() for d in ts.index])
    elif kind == 'holiday':
        holiday_set = set(day_holiday)
        day_enconding_feature = np.array([1 if d in holiday_set else 0 for d in ts.index])
    elif kind == 'spring_festival':
        spring_festival_set = set(day_spring_festival)
        day_enconding_feature = np.array([1 if d in spring_festival_set else 0 for d in ts.index])
    elif kind == 'promotion':
        promotional_activities_set = set(day_promotion)
        day_enconding_feature = np.array([1 if d in promotional_activities_set else 0 for d in ts.index])
        
    return day_enconding_feature
    
        
def day_moving_window_feature_onehot(day_enconding_feature,look_back,onehot=False):
    '''
    将日期类型做onehot处理
    '''
    encoder = LabelBinarizer()
    day_enconding_feature = day_enconding_feature.reshape(-1,1)
    dataY = []
    for i in range(len(day_enconding_feature)-look_back):
        dataY.append(day_enconding_feature[i + look_back,0])
    if onehot == True:
        day_enconding_feature_onehot = encoder.fit_transform(np.array(dataY))
    else:
        day_enconding_feature_onehot = np.array(dataY).reshape(-1,1)
    return day_enconding_feature_onehot

def ts_moving_window_feature(ts, look_back):
    '''
    滑窗操作
    '''
    dataset = ts.values.reshape(-1,1)
    dataset = dataset.astype('float32')
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    X = np.array(dataX)
    y = np.array(dataY)
    return X,y


def prepare_predict_feature(x,day,day_week = False,day_holiday=False,day_spring_festival=False,day_promotion=False):
    '''
    对未来的日期作同样的编码操作
    '''
    if day_week == False:
        pass
    else:
        w = np.zeros(7)
        w[day.weekday()]=1
        x = np.hstack([x,w])
    
    if day_holiday == False:
        pass
    else:
        holiday_set = set(day_holiday)
        h = [1 if d in holiday_set else 0 for d in [day]]
        x = np.hstack([x,h])

    if day_spring_festival == False:
        pass
    else:
        spring_festival_set = set(day_spring_festival)
        sf = [1 if d in spring_festival_set else 0 for d in [day]]
        x = np.hstack([x,sf])
        
    if day_promotion == False:
        pass
    else:
        promotional_activities_set = set(day_promotion)
        pa = [1 if d in promotional_activities_set else 0 for d in [day]]
        x = np.hstack([x,pa])
        
    x = x.reshape(1,-1)
    return x

def move_window_model(ts,train_period,predict_period,look_back,day_week = False,day_holiday=False,day_spring_festival=False,day_promotion=False,draw=False): 
    '''
    基于日期编码和滑窗的模型
    参数说明：
    ts : 输入历史时间序列
    train_period ：训练数据的时间区间  比如 ['2017-3-1','2017-5-10']
    predict_period : 预测的时间区间 比如 ['2017-5-10','2017-6-20']
    # 注意 train_period[1] 和 predict_period[0] 要相同
    look_back : 滑窗时窗口选择的大小
    day_week : 取值为 False or True , 用于决定是否将星期信息编码
    day_holiday : 取值为 False 或者 时间对象的list
    day_spring_festival ： 取值为 False 或者 时间对象的list
    day_promotion ： 取值为 False 或者 时间对象的list
    draw : 是否将结果绘制出来，服务器上跑的时候默认为False
    函数返回预测时间段的一个时间序列

    备注：其中的LinearModel可以替换为其他的回归模型，比如XGBoost等，但是为了稳定性和减少模型参数，此处采用LinearModel
    '''
    ts_train = ts[train_period[0]:train_period[1]]
    ts_predict = pd.Series(index = pd.date_range(predict_period[0],predict_period[1]))
    
    X,y = ts_moving_window_feature(ts_train,look_back)
    
    # Add More Feature
    
    if day_week == False:
        pass
    else:
        week_feature = day_enconding(ts_train,kind='week')
        week_feature_onehot = day_moving_window_feature_onehot(week_feature,look_back,onehot=True)
        X = np.hstack([X,week_feature_onehot])
    
    if day_holiday == False:
        pass
    else:
        holiday_feature = day_enconding(ts_train,kind='holiday',day_holiday = day_holiday)
        holiday_feature_onehot = day_moving_window_feature_onehot(holiday_feature,look_back,onehot=False)
        X = np.hstack([X,holiday_feature_onehot])

    if day_spring_festival == False:
        pass
    else:
        spring_festival_feature = day_enconding(ts_train,kind='spring_festival',day_spring_festival= day_spring_festival)
        spring_festival_feature_onehot = day_moving_window_feature_onehot(spring_festival_feature,look_back,onehot=False)
        X = np.hstack([X,spring_festival_feature_onehot])
        
    if day_promotion == False:
        pass
    else:
        promotion_feature = day_enconding(ts_train,kind='promotion',day_promotion=day_promotion)
        promotion_feature_onehot = day_moving_window_feature_onehot(promotion_feature,look_back,onehot=False)
        X = np.hstack([X,promotion_feature_onehot])   
    
    LinearModel = LinearRegression()
    LinearModel.fit(X,y)
    
    
    start = pd.to_datetime(predict_period[0])-datetime.timedelta(days = look_back)
    end = pd.to_datetime(predict_period[0])-datetime.timedelta(days = 1)
    x = ts[start:end].values
    
    perdict_day_range = ts_predict.index
    y_predict = []
    
    for day in perdict_day_range:
        x_for_model = prepare_predict_feature(x,day,day_week,day_holiday,day_spring_festival,day_promotion)
        y = LinearModel.predict(x_for_model)[0]
        y_predict.append(y)
        x = np.hstack([x[1:],[y]])
    y_predict = pd.Series(y_predict,index = perdict_day_range)
    
    if draw == True:        
        ts[train_period[0]:train_period[1]].plot()
        ts[predict_period[0]:predict_period[1]].plot()
        y_predict.plot()
        plt.legend(['train','test','predict'])
    
    return y_predict
   
'''    
from fbprophet import Prophet
def predict_ts(ts,train_period,predict_period,draw = False):
    ts_train = ts[train_period[0]:train_period[1]]
    ts_predict = pd.Series(index = pd.date_range(predict_period[0],predict_period[1]))
    
    m = Prophet()
    df = pd.DataFrame()
    df['y'] = ts_train
    df['ds'] = ts_train.index
    m.fit(df)
    future = m.make_future_dataframe(periods=len(ts_predict)-1) 
    forecast = m.predict(future)
    pred_ts = pd.Series(forecast.yhat.values,index = forecast.ds)[predict_period[0]:predict_period[1]]
    if draw == True:        
        ts[train_period[0]:train_period[1]].plot()
        ts[predict_period[0]:predict_period[1]].plot()
        pred_ts.plot()
        plt.legend(['train','test','predict'])
    return pred_ts


def arma_predict(ts,train_period,predict_period,p=7,q=3,draw=False):
    from statsmodels.tsa.arima_model import ARMA
    ts = ts.astype('float')
    ts_train = ts[train_period[0]:train_period[1]]
    arma_mod = ARMA(ts_train,(p,q)).fit()
    Predict = arma_mod.predict(start=predict_period[0],end=predict_period[1])
    if draw == True:        
        ts[train_period[0]:train_period[1]].plot()
        ts[predict_period[0]:predict_period[1]].plot()
        Predict.plot()
        plt.legend(['train','test','predict'])
    return Predict

'''    

def trend_period_model(ts,train_period,predict_period,filter_up=0.9,filter_down=0.1,draw=False):
    '''
    将趋势和周期因素分离预测的模型
    参数说明：
    filter_up ： 过滤异常时的百分位上限
    filter_down ： 过滤异常时的百分位下限
    '''
    filter_up = 0.9
    filter_down = 0.1
    ts_train = ts[train_period[0]:train_period[1]]
    ts_predict = pd.Series(index = pd.date_range(predict_period[0],predict_period[1]))
    model,_ = linear_fun_fit(ts_filter(ts_train,filter_up,filter_down))
    X_p = np.arange(len(ts_train)+len(ts_predict)-1)[:,np.newaxis]
    ts_p = pd.Series(model.predict(X_p).reshape(1,-1)[0],index=pd.date_range(train_period[0],predict_period[1]))


    df_ts=pd.DataFrame()
    data_for_week = ts_train[ts_train / ts_filter(ts_train,filter_up,filter_down) == 1]
    df_ts['value'] = data_for_week
    df_ts['week'] = [d.weekday() for d in data_for_week.index]
    week_period = df_ts.pivot_table('value','week',aggfunc='mean')
    week_ratio = week_period/week_period.mean()

    ts_pd = pd.DataFrame()
    ts_pd['value'] = ts_p
    ts_pd['week'] = [d.weekday() for d in ts_p.index]
    try:
        ts_pd['new_value'] = [v*week_ratio.loc[w,:].values[0] for v,w in zip(ts_pd.value,ts_pd.week)]
    except:        
        ts_pd['new_value'] = [v*week_ratio[w] for v,w in zip(ts_pd.value,ts_pd.week)]
    ts_pd_p = ts_pd.new_value
    predict = ts_pd_p[predict_period[0]:predict_period[1]]
    if draw == True:        
        ts[train_period[0]:train_period[1]].plot()
        ts[predict_period[0]:predict_period[1]].plot()
        predict.plot()
        plt.legend(['train','test','predict'])
    return predict

def error(real,pred):
    m = real.mean()
    e = (abs(real-pred)/m).mean()
    return e    
    
def combine_model(ts,yesterday,predict_days,day_week,day_holiday,day_spring_festival,day_promotion,draw=False):
    '''
    对几个基础模型进行融合
    yesterday ： 预测开始的时间，模型中默认是昨天
    predict_days ： 预测是时常
    day_week : 取值为 False or True , 用于决定是否将星期信息编码
    day_holiday : 取值为 False 或者 时间对象的list
    day_spring_festival ： 取值为 False 或者 时间对象的list
    day_promotion ： 取值为 False 或者 时间对象的list
    '''

    ts_backup = ts[:]

    short_long_split = 60 # 该值为区分长期预测和短期预测的分界值，默认60天一下为短期预测，60天以上为长期预测

    if predict_days > short_long_split:
        short_or_long = 'long'
    else:
        short_or_long = 'short'
        
    print '\nconsider as %s term prediction'%short_or_long
    
    predict_period = [yesterday,yesterday+timedelta(predict_days)]    
    pd_yesterday = pd.datetime(yesterday.year,yesterday.month,yesterday.day)

    short_train_days = 120 # 短期预测使用预测当日往前的 short_train_days 这么多天数据
    long_train_days = min(210,len(ts)-20) # 长期预测使用预测当日往前的 long_train_days 这么多天数据

    test_days = 30 # 模型融合时有个参照权重是依据验证集合计算的，取最近的 test_days 天作验证

    if short_or_long == 'short':
        train_period = [yesterday-timedelta(short_train_days),yesterday]
        test_train_period = [train_period[0]-timedelta(test_days) ,train_period[1]-timedelta(test_days)]
        test_predict_period = [predict_period[0]-timedelta(test_days),predict_period[1]-timedelta(test_days)]
        look_back = 30 
    if short_or_long == 'long' or (pd_yesterday in spring_festival):
        train_period = [yesterday-timedelta(long_train_days),yesterday]
        test_train_period = [train_period[0]-timedelta(test_days) ,train_period[1]-timedelta(test_days)]
        test_predict_period = [predict_period[0]-timedelta(test_days),predict_period[1]-timedelta(test_days)]
        look_back = 90
        
    sf = np.array([pd.datetime(2017,1,28),pd.datetime(2018,2,15),pd.datetime(2019,2,4)]) # 对靠近春节的日期作特殊标记
    close_to_spring_festive = np.array([x.days for x in train_period[1] - sf])

    if ((close_to_spring_festive>-10) & (close_to_spring_festive<look_back)).any():
        ts = ts_filter(ts,1,0.3)

    # 在验证集上计算误差
    a = move_window_model(ts,test_train_period,test_predict_period,look_back,day_week,day_holiday,day_spring_festival,day_promotion,draw=False)
    b = trend_period_model(ts,test_train_period,test_predict_period,draw=False)
    real = ts[test_predict_period[0]:test_predict_period[1]]
    errors = [error(real,a),error(real,b)]

    # 使用多个模型对未来进行预测
    a = move_window_model(ts,train_period,predict_period,look_back,day_week,day_holiday,day_spring_festival,day_promotion,draw=False)
    b = trend_period_model(ts,train_period,predict_period,draw=False)

    # 融合多个模型
    c_e = errors[1]*1.0/sum(errors) * a + errors[0]*1.0/sum(errors) * b
    c_short = 0.8 * a + 0.2 * b  # short-term
    c_long = 0.2 * a + 0.8 * b # long-term
    if short_or_long == 'short':
        c = (c_e+c_short)/2.0
    else:
        c = (c_e+c_long)/2.0
        
    sf = np.array([pd.datetime(2017,1,28),pd.datetime(2018,2,15),pd.datetime(2019,2,4)])
    close_to_spring_festive = np.array([x.days for x in train_period[1] - sf])

    # 对超长期的预测作修正，长期而言，趋势周期更稳定
    if predict_days > 180 :
        c = pd.concat([c[:150],b[150:]])
    
    if draw == True:        
        ts_backup[train_period[0]:train_period[1]].plot()
        ts_backup[predict_period[0]:predict_period[1]].plot()
        c.plot()
        plt.legend(['train','test','predict'])
    return c,train_period
    
holiday = [pd.datetime(2016,1,1),
           pd.datetime(2016,1,2),
           pd.datetime(2016,1,3),
           pd.datetime(2016,4,1),
           pd.datetime(2016,4,2),
           pd.datetime(2016,4,3),
           pd.datetime(2016,4,4),
           pd.datetime(2016,4,29),
           pd.datetime(2016,4,30),
           pd.datetime(2016,5,1),
           pd.datetime(2016,5,2),
           pd.datetime(2016,6,8),
           pd.datetime(2016,6,9),
           pd.datetime(2016,6,10),
           pd.datetime(2016,6,11),
           pd.datetime(2016,9,14),
           pd.datetime(2016,9,15),
           pd.datetime(2016,9,16),
           pd.datetime(2016,9,17),
           pd.datetime(2016,9,30),
           pd.datetime(2016,10,1),
           pd.datetime(2016,10,2),
           pd.datetime(2016,10,3),
           pd.datetime(2016,10,4),
           pd.datetime(2016,10,5),
           pd.datetime(2016,10,6),
           pd.datetime(2016,10,7),
           pd.datetime(2016,12,31),
           pd.datetime(2017,1,1),
           pd.datetime(2017,3,31),
           pd.datetime(2017,4,1),
           pd.datetime(2017,4,2),
           pd.datetime(2017,4,3),
           pd.datetime(2017,4,4),
           pd.datetime(2017,4,28),
           pd.datetime(2017,4,29),
           pd.datetime(2017,4,30),
           pd.datetime(2017,5,1),
           pd.datetime(2017,5,27),
           pd.datetime(2017,5,28),
           pd.datetime(2017,5,29),
           pd.datetime(2017,5,30),
           pd.datetime(2017,9,30),
           pd.datetime(2017,10,1),
           pd.datetime(2017,10,2),
           pd.datetime(2017,10,3),
           pd.datetime(2017,10,4),
           pd.datetime(2017,10,5),
           pd.datetime(2017,10,6),
           pd.datetime(2017,10,7),
           pd.datetime(2017,10,8),
          ]

spring_festival = [pd.datetime(2016,2,5),
                   pd.datetime(2016,2,6),
                   pd.datetime(2016,2,7),
                   pd.datetime(2016,2,8),
                   pd.datetime(2016,2,9),
                   pd.datetime(2016,2,10),
                   pd.datetime(2016,2,11),
                   pd.datetime(2016,2,12),
                   pd.datetime(2016,2,13),
                   pd.datetime(2017,1,27),
                   pd.datetime(2017,1,28),
                   pd.datetime(2017,1,29),
                   pd.datetime(2017,1,30),
                   pd.datetime(2017,1,31),
                   pd.datetime(2017,2,1),
                   pd.datetime(2017,2,2),
                   pd.datetime(2017,2,2),
                   ]

promotional_activities = [pd.datetime(2017,3,31),
                          pd.datetime(2017,4,28),
                          pd.datetime(2017,5,26),
                          pd.datetime(2017,6,30),
                          pd.datetime(2017,7,28),
                          pd.datetime(2017,8,25),
                          pd.datetime(2017,9,29),
                          pd.datetime(2017,10,27),
                          pd.datetime(2017,11,24),
                          pd.datetime(2017,12,29)
                         ]

#------------------------------------






# -------------read data-------------



today = pd.to_datetime(time.strftime('%Y-%m-%d',time.localtime(time.time())))
yesterday= today - timedelta(1) # 如果线下数据没有更新到昨天那么这个1得改，使得yesterday 时线下测试时数据的最大日期


day_week = True
day_holiday = holiday
day_spring_festival = spring_festival
day_promotion = promotional_activities
predict_days = input('predict days : ')



#------------- process data -------------

#data= pd.read_excel('data_test.xlsx',parse_dates=['time']) # 线下
data = pd.read_csv('ChinaFinishOrder.csv','\t',parse_dates=['time']) # 线上
data.index = data.time
del data['time']
#ts = data.finish # 线下
#ts = ts[ts.index>'2016-5-1'] # 线下
ts = data.finish_order_cnt # 线上

ts_pred,train_period =combine_model(ts,yesterday,predict_days,day_week,day_holiday,day_spring_festival,day_promotion,draw=False)
ts_pred = ts_pred[1:].astype('int')
ts_past_real = ts[train_period[0]:train_period[1]]
#print ts_past_real[-5:]
#print ts_pred[:5]

ts_combine = pd.concat([ts_past_real,ts_pred])
ts_combine_week = ts_combine.resample('W').mean()
#ts_combine_week.plot()

result_real = pd.DataFrame({'time':[str(d.date()) for d in ts_past_real.index],'v':ts_past_real.values,'type':['past']*len(ts_past_real)})
result_pred = pd.DataFrame({'time':[str(d.date()) for d in ts_pred.index],'v':ts_pred.values,'type':['pred']*len(ts_pred)})
result = pd.concat([result_real,result_pred]).reset_index(drop=1)

result.to_csv('Predict_ts.csv',index=False,header=True)
