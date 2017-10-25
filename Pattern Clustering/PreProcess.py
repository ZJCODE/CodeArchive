#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:21:19 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np

def Remove_Event_Influence():
    pass



def Get_Stock_Ts(data_df,code,filed='S_DQ_ADJCLOSE'):
    '''
    返回某只股票的信息
    
    ---------------
    filed :
        
    S_INFO_WINDCODE,
    TRADE_DT,
    S_DQ_VOLUME, 
    S_DQ_ADJPRECLOSE,
    S_DQ_ADJOPEN, 
    S_DQ_ADJHIGH' 
    S_DQ_ADJLOW,
    S_DQ_ADJCLOSE,
    S_DQ_PCTCHANGE
    ---------------
    
    '''
    ts = data_df[data_df.S_INFO_WINDCODE == code][filed]
    ts = ts.sort_index()
    return ts


def Filter_Nan_For_Several_Ts(ts_list):
    '''
    多个对应的时间序列，如果某天某个对象某一天没有数据，那么把所有对象的这一天的数据都去除
    
    -----------------------------------
    Example
    
    ts_list :
        [
         2017-05-24    0.0
         2017-05-25    1.0
         2017-05-26    2.0
         2017-05-31    3.0
         2017-06-01    4.0
         2017-06-02    5.0
         2017-06-05    6.0
         2017-06-06    7.0
         2017-06-07    8.0
         2017-06-08    9.0
        ,
         2017-05-24    2.0
         2017-05-25    3.0
         2017-05-26    4.0
         2017-05-31    NaN
         2017-06-01    4.0
         2017-06-02    5.0
         2017-06-05    NaN
         2017-06-06    4.0
         2017-06-07    3.0
         2017-06-08    2.0
        ]    
        
        
    Filter_Nan_For_Several_Ts(ts_list) :
        
        [2017-05-24    0.0
         2017-05-25    1.0
         2017-05-26    2.0
         2017-06-01    4.0
         2017-06-02    5.0
         2017-06-06    7.0
         2017-06-07    8.0
         2017-06-08    9.0
         ,
         2017-05-24    2.0
         2017-05-25    3.0
         2017-05-26    4.0
         2017-06-01    4.0
         2017-06-02    5.0
         2017-06-06    4.0
         2017-06-07    3.0
         2017-06-08    2.0
         ]
        
    -----------------------------------
    
    '''
    ts_list_processd = []
    # 避免多个时间序列时间长短不同，先获取全部序列的最小时间和最大时间
    max_time = ts_list[0].index[0]
    min_time = ts_list[0].index[0]
    for ts in ts_list:
        max_ts_temp = max(ts.index)
        min_ts_temp = min(ts.index)
        if max_ts_temp > max_time:
            max_time = max_ts_temp
        if min_ts_temp < min_time:
            min_time = min_ts_temp
    # 将几个时间序列合并到一个dataframe中，然后将含有nan的行drop掉
    ts_combine = pd.DataFrame(index = pd.date_range(min_time,max_time,freq='D'))
    for i,ts in enumerate(ts_list):
        ts_combine[str(i)] = ts
    ts_combine = ts_combine.dropna()
    for i in range(len(ts_list)):
        ts_list_processd.append(ts_combine[str(i)])
    return ts_list_processd


def Prepare_Ts_DataFrame(data_df,code_list,filed):
    '''
    将数据组织成每一列为一只股票信息的格式，方便读取，从而节省检索时间


    filed :
        
    S_INFO_WINDCODE,
    TRADE_DT,
    S_DQ_VOLUME, 
    S_DQ_ADJPRECLOSE,
    S_DQ_ADJOPEN, 
    S_DQ_ADJHIGH' 
    S_DQ_ADJLOW,
    S_DQ_ADJCLOSE,
    S_DQ_PCTCHANGE

    
    -----------------------------------
    Example
    
        
               S_INFO_WINDCODE    TRADE_DT  S_DQ_VOLUME  S_DQ_ADJCLOSE
    TRADE_DT                                                         
    2007-02-02       600781.SH    2007-02-02     13426.66          10.78
    2007-02-02       600853.SH    2007-02-02    101885.57           6.44
    2007-02-02       600862.SH    2007-02-02     57897.99          14.86
    2007-02-02       600881.SH    2007-02-02    482970.94         190.23
    2007-02-02       600892.SH    2007-02-02      3718.00           7.59
    ...			    ...	 		...		      ...             ...	
    
    转变为包含类似如下 dataframe 的 list
    
                600781.SH  600853.SH  600862.SH  600881.SH  600892.SH       ...
    2007-02-02      10.78       6.44      14.86     190.23       7.59       ... 
    2007-02-03        NaN        NaN        NaN        NaN        NaN       ...
    2007-02-04        NaN        NaN        NaN        NaN        NaN       ...
    2007-02-05      10.72       6.50      14.12     197.75       7.49       ...
    2007-02-06      10.76       6.89      14.34     196.08       7.60       ...
    2007-02-07      10.87       6.98      14.47     198.59       7.64       ...  
    2007-02-08      11.42       7.68      14.38     197.19       8.01       ...
    ...             ...         ...        ...       ...           ...      ...
    
    -----------------------------------
    
    '''
    max_time = pd.to_datetime(max(data_df.TRADE_DT.unique())).date()
    min_time = pd.to_datetime(min(data_df.TRADE_DT.unique())).date()
    df_list = []
    for f in filed:
        df = pd.DataFrame(index = pd.date_range(min_time,max_time,freq='D'))
        for code in code_list:
            print code
            df[code] = Get_Stock_Ts(data_df,code,filed=f)
        df_list.append(df)
    return df_list




