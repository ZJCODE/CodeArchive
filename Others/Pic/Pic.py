#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:14:07 2016

@author: ZJun
"""
import seaborn
import scipy.io as sio
import matplotlib.pyplot as plt

# load data
data=sio.loadmat('matlab.mat')
X = data['X'][0]
Ycon = data['Ycon'].flatten() / 2000.0
Ylin = data['Ylin'].flatten() / 2000.0

#plot
seaborn.set()
plt.rc('figure',figsize=(16,11))
plt.plot(X[2:12:2],Ycon[2:12:2])
plt.plot(X[2:12:2],Ylin[2:12:2])
plt.legend(['Constant Assumption','Linear Auusmption'],loc='upper left',fontsize=15)
plt.xlabel('Local Windows Size',fontsize=14)
plt.ylabel('Average Error',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()