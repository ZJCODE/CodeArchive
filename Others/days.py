# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:54:22 2016

@author: ZJun
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_table('data.txt')

plt.rc('figure',figsize = (18,5))

index = pd.date_range(start='2016-9-10',end='2016-11-10',freq = 'D')

ts = pd.Series(data.c.values,index = index)


r = 0.1
#ts.plot(kind='bar',alpha=0.4)

up = ts[-4]*(1+r)
down = ts[-4]*(1-r)

ts[(ts.values>down) & (ts.values <up)].mean()

plt.plot(range(len(ts)),np.ones(len(ts))*up,'r')
plt.plot(range(len(ts)),np.ones(len(ts))*down,'g')
plt.bar(range(len(ts)),ts.values,alpha=0.4)

ts.plot(kind='bar',alpha=0.4)
plt.ylim([3,5])



from statsmodels.tsa.arima_model import ARMA
arma_mod = ARMA(ts,(2,2)).fit()
Predict = arma_mod.predict(start='2016-9-10',end='2016-11-13')



r = 0.03
#ts.plot(kind='bar',alpha=0.4)

up = Predict['2016-11-11']*(1+r)
down = Predict['2016-11-11']*(1-r)

plt.plot(range(len(ts)),np.ones(len(ts))*up,'r')
plt.plot(range(len(ts)),np.ones(len(ts))*down,'g')
plt.bar(range(len(ts)),ts.values,alpha=0.4)

plt.ylim([3,5])



Predict.plot(kind='bar',alpha=0.4)
#ts.plot()
plt.ylim([3,5])


Predict.plot()
ts.plot()