#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:26:29 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

def Modify_Buy_Sell_Action(act):
    '''
    去除连续购买和连续卖出的后序操作
    '''
    act_modify = [act[0]]
    flag = [act[0]]
    for i in range(1,len(act)):
        if act[i] == 0 or act[i] == flag :
            act_modify.append(0)
        else:
            act_modify.append(act[i])
            flag = act[i]
    return np.array(act_modify)



def Draw_Buy_Sell(act,ts):
    '''
    将买卖操作可视化绘制出来
    '''
    import datetime
    tomorrow = pd.Series(ts[-1],index = [ts.index[-1]+ datetime.timedelta(days=1)])
    ts_ = ts.append(tomorrow)

    act_buy = np.array([i if i > 0 else 0 for i in act])*ts_
    act_buy = act_buy[act_buy>0]
    act_sell = np.array([i if i < 0 else 0 for i in act])*ts_
    act_sell = act_sell[act_sell<0]*-1

    plt.rc('figure',figsize=[20,10])
    plt.plot(ts_,alpha=0.3)
    plt.plot(act_buy.index,act_buy.values,'^r',markersize=8)
    plt.plot(act_sell.index,act_sell.values,'vg',markersize=8)
    plt.legend(['close Index','buy','sell'],fontsize=13)
    plt.show()

    return act_buy,act_sell



def Profit_Evaluation(act,ts,ts_pct_change,kind = 's'):
    '''
    收益率评价
    '''
    act_ts = pd.Series(act[:-1],index=ts.index)
    profit = []
    buy = 0
    sell = 0
    flag =0 
    buy_flag = 0
    
    for i in range(len(act_ts)):
        day = ts.index[i]
        if flag == 0 and act_ts[i] == -1:
            profit.append(0)
            flag = 1
            continue
        elif act_ts[i] == 1 and ts_pct_change[day] < 9.99:
            buy = ts[i]
            profit.append(0)
            flag = 1
            buy_flag = 1
        elif act_ts[i] == -1 and ts_pct_change[day] < 9.99:
            sell = ts[i]
            if kind == 's':
            	profit.append(0.9985*(sell-buy)/buy) 
            elif kind== 'f':
            	profit.append(0.9995*(sell-buy)/buy) 
            else:
            	profit.append((sell-buy)/buy) 
            buy_flag = 0
        else:
            profit.append(0)
    
    cumprod_profit = np.cumproduct(np.array(profit)+1)
    profit_ts = pd.Series(profit,ts.index)
    cumprod_profit_ts = pd.Series(cumprod_profit,ts.index)


    buy_flag = 0
    for i in range(len(cumprod_profit_ts)):
        if act[i] == 1:
            buy_flag = 1
        if act[i] == -1:
            buy_flag = 0
        if buy_flag == 1 :
            cumprod_profit_ts[i] = cumprod_profit_ts[i-1] * (ts[i]/ts[i-1])
    
    return profit_ts,cumprod_profit_ts

