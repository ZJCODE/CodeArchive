# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:31:16 2016

@author: Admin
"""
import pandas as pd
from datetime import timedelta
from datetime import datetime

NFS_Daily = pd.read_excel('.//OFFLINE_20160618//NFS_Daily.xlsx')
NFS_Weekly = pd.read_excel('.//OFFLINE_20160618//NFS_Weekly.xlsx')
#--------------------------------------------------------

NSO_Daily = pd.read_excel('.//OFFLINE_20160618//NSO_Daily.xlsx')
NSO_Weekly = pd.read_excel('.//OFFLINE_20160618//NSO_Weekly.xlsx')

#---------------------------------------------------------
'''
NFS_Date = NFS_Daily.TRAN_DT

NFS_Week  = [date - timedelta(date.weekday()) for date in NFS_Date] 

NFS_Daily['TRAN_DT'] = NFS_Week
NFS_Daily.columns =['WEEK_DESCRIPTION', 'DIVISION','RETL_GNDR_GRP','STORE_NM','STORE_TYPE','STORE_CITY','City','SLS_USD','SLS_QTY']


d1 = datetime(2014,9,29)
d2 = datetime(2015,2,23)

NFS_Daily_P1 = NFS_Daily[NFS_Daily.WEEK_DESCRIPTION >= d1]
NFS_Daily_P1 = NFS_Daily_P1[NFS_Daily_P1.WEEK_DESCRIPTION <= d2]

d3 = datetime(2015,9,28)
NFS_Daily_P2 = NFS_Daily[NFS_Daily.WEEK_DESCRIPTION >= d3]


NFS_ALL_Week = pd.concat([NFS_Daily_P1,NFS_Daily_P2,NFS_Weekly])

NFS_ALL_Week.to_csv('NFS_ALL_Week.csv',encoding='gb2312',header=True,index = False)

'''

#--------------------------------------------------

NSO_Date = NSO_Daily.TRAN_DT

NSO_Week  = [date - timedelta(date.weekday()) for date in NSO_Date] 

NSO_Daily['TRAN_DT'] = NSO_Week
NSO_Daily.columns =['WEEK_DESCRIPTION', 'DIVISION','RETL_GNDR_GRP','STORE_NM','STORE_TYPE','STORE_CITY','City','SLS_USD','SLS_QTY']


d1 = datetime(2014,9,29)
d2 = datetime(2015,2,23)

NSO_Daily_P1 = NSO_Daily[NSO_Daily.WEEK_DESCRIPTION >= d1]
NSO_Daily_P1 = NSO_Daily_P1[NSO_Daily_P1.WEEK_DESCRIPTION <= d2]

d3 = datetime(2015,9,28)
NSO_Daily_P2 = NSO_Daily[NSO_Daily.WEEK_DESCRIPTION >= d3]


NSO_ALL_Week = pd.concat([NSO_Daily_P1,NSO_Daily_P2,NSO_Weekly])

NSO_ALL_Week.to_csv('NSO_ALL_Week.csv',encoding='gb2312',header=True,index = False)
