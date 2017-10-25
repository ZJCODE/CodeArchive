#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:54:11 2017

@author: zhangjun
"""

import os.path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PreProcess import Get_Stock_Ts
from Pattern_Toolkit import Extract_Pattern_and_Target_With_Resample
from Pattern_Toolkit import Pattern_Cluster,Profit_Judge
import gc
import seaborn
seaborn.set()


# 读取股票数据，可以读存到本地的数据，也可以直接连数据库读取
data_df = pd.read_csv('data.csv',parse_dates=['TRADE_DT'])
data_df.index = data_df.TRADE_DT

# 随机抽取一部分股票用于测试,之后可以替换成自己想看的股票池代码[code_pool]
all_code = data_df.S_INFO_WINDCODE.unique()
np.random.shuffle(all_code)
code_pool = all_code[:10]
gc.collect()

#stock = Prepare_Ts_DataFrame(data_df,code_pool,filed=['S_DQ_ADJCLOSE','S_DQ_VOLUME','S_DQ_PCTCHANGE'])

# 模式序列长度
pattern_len = input('Enter The Length of Pattern : ')
# 模式后序序列长度
target_len = input('Enter The Length of Target : ')
# 序列 rolling mean 的长度
resample_list = [2,5,10,20]

# 初始化模式库
pattern_list = np.ones(pattern_len)
target_list = np.ones(target_len+1)


# 从整个股票池抽取模式库
print 'Extract Pattern '

for code in code_pool:
    print code
    # 获取该股票代码的时间序列
    ts_p = Get_Stock_Ts(data_df,code,filed='S_DQ_ADJCLOSE')
    # 抽取该时间序列的模式和其后序序列
    pattern,target = Extract_Pattern_and_Target_With_Resample(ts_p,pattern_len,target_len,resample_list,normalization=True)
    # 将抽取的模式加入到整个模式库中
    pattern_list = np.vstack([pattern_list,pattern])
    target_list = np.vstack([target_list,target])


# 将模式库存到本地，方便之后重复使用，不需要再提取

np.save('pattern_list.npy',pattern_list)
np.save('target_list.npy',target_list)