# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:44:55 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

import glob
Files = glob.glob(r'./Data/*.csv')

mfd_bank_shibor = pd.read_csv('./Data/mfd_bank_shibor.csv')
mfd_day_share_interest = pd.read_csv('./Data/mfd_day_share_interest.csv')
user_balance_table = pd.read_csv('./Data/user_balance_table.csv')
user_profile_table = pd.read_csv('./Data/user_profile_table.csv')

mfd_bank_shibor.mfd_date = [pd.to_datetime(str(date)) for date in mfd_bank_shibor.mfd_date]
mfd_day_share_interest.mfd_date = [pd.to_datetime(str(date)) for date in mfd_day_share_interest.mfd_date]
user_balance_table.report_date = [pd.to_datetime(str(date)) for date in user_balance_table.report_date]




total_purchase_redeem_amt = user_balance_table[['user_id','report_date','total_purchase_amt','total_redeem_amt']]
Redeem = total_purchase_redeem_amt.pivot_table('total_redeem_amt','report_date',aggfunc = 'sum')
Purchase = total_purchase_redeem_amt.pivot_table('total_purchase_amt','report_date',aggfunc = 'sum')

Redeem.plot()
Pre_Redeem.plot()
Purchase.plot()
Pre_Purchase.plot()

from statsmodels.tsa.ar_model import AR
ar_mod = AR(Redeem)
ARM = ar_mod.fit(10)
#AR_FIT =ARM.fittedvalues
Pre_Redeem= ARM.predict(10,456)[datetime(2014,9,1):datetime(2014,9,30)]


ar_mod = AR(Purchase)
ARM = ar_mod.fit(10)
#AR_FIT =ARM.fittedvalues
Pre_Purchase= ARM.predict(10,456)[datetime(2014,9,1):datetime(2014,9,30)]



#--------------------------------------------------------------------------

Pre_Data = pd.DataFrame({'Date':Pre_Redeem.index})
Pre_Data.loc[:,'Pre_Purchase'] = [int(i) for i in Pre_Purchase.values]
Pre_Data.loc[:,'Pre_Redeem'] = [int(i) for i in Pre_Redeem.values]

def Change_Date_Formate(Date):
    Month = [str(i.month) if i.month>=10 else str(0)+str(i.month) for i in [Date]][0]
    Day = [str(i.day) if i.day>=10 else str(0)+str(i.day) for i in [Date]][0]
    return int(str(Date.year)+Month+Day)
    
Date = [Change_Date_Formate(date) for date in Pre_Data.Date]
Pre_Data.loc[:,'Date'] = Date

Pre_Data.to_csv('tc_comp_predict_table.csv',index=False,header=False)