#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:13:53 2017

@author: zhangjun
"""

import cx_Oracle
import pandas as pd


conn =cx_Oracle.connect("wind_read_only/wind_read_only@192.168.1.192:1521/ORCL")
c=conn.cursor()


start_day = 20070201



# 股票
SQL = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_VOLUME,S_DQ_ADJPRECLOSE,S_DQ_ADJOPEN,S_DQ_ADJHIGH,S_DQ_ADJLOW,S_DQ_ADJCLOSE,S_DQ_PCTCHANGE \
        from wind.ASHAREEODPRICES \
        where TRADE_DT > %d and substr(S_INFO_WINDCODE,8)= 'SH' "%(start_day)

c.execute(SQL)
data = c.fetchall()
columns = ['S_INFO_WINDCODE','TRADE_DT','S_DQ_VOLUME','S_DQ_ADJPRECLOSE','S_DQ_ADJOPEN','S_DQ_ADJHIGH','S_DQ_ADJLOW','S_DQ_ADJCLOSE','S_DQ_PCTCHANGE']
data_df = pd.DataFrame(data,columns=columns)

data_df.index = [pd.to_datetime(d).date() for d in data_df.TRADE_DT.values]

data_df.to_csv('data.csv',index=False)


# 指数


SQL = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_VOLUME,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE,S_DQ_PCTCHANGE \
        from wind.AINDEXEODPRICES \
        where TRADE_DT > %d and S_INFO_WINDCODE in ('000300.SH','000905.SH','000001.SH')"%(start_day)

c.execute(SQL)
data = c.fetchall()
columns = ['S_INFO_WINDCODE','TRADE_DT','S_DQ_VOLUME','S_DQ_PRECLOSE','S_DQ_OPEN','S_DQ_HIGH','S_DQ_LOW','S_DQ_CLOSE','S_DQ_PCTCHANGE']
data_df = pd.DataFrame(data,columns=columns)

data_df.index = [pd.to_datetime(d).date() for d in data_df.TRADE_DT.values]

data_df.to_csv('data_index.csv',index=False)



