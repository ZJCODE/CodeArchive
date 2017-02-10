#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:32:15 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from twitter_function import Import_Obj,GenerateDate,hinton,GetPartOfTimeSeries,Sort_Dict


population_2015 = dict(zip(['Cairns','Townsville','Mackay','Sunshine Coast','Brisbane','Gold Coast'],
                           [147993,180333,85455,302122,2209453,624918]))

     
P = Sort_Dict(population_2015)
Places = [i[0] for i in P]

df_where_to_where = Import_Obj('./DF_Result/df_where_to_where')
actual_days_in_week = Import_Obj('./Data/actual_days_in_week')
df_real_flu = Import_Obj('./DF_Result/df_real_flu')[population_2015.keys()]
df_twitter_in_place_loc = Import_Obj('./DF_Result/df_twitter_in_place_loc')[population_2015.keys()]

'''
Queensland_Flu = pd.read_csv('./Data/Queensland2015.csv')
from datetime import timedelta
def GetRealFlu(Queensland_Flu,location=None):
    first_week = pd.datetime(2015,1,5).date()
    t = [first_week]
    if location != None:        
        x = Queensland_Flu[location].values
    else:
        x = Queensland_Flu.ix[:,1:].sum(1).values
    for i in range(1,len(x)):
        t.append(first_week + timedelta(i*7))
    ts = pd.Series(x,index = t[:len(x)])
    return ts

real_flu = GetRealFlu(Queensland_Flu,Places[0]).values
index = GetRealFlu(Queensland_Flu,Places[0]).index
for p in Places[1:]:
    real_flu = np.c_[real_flu,GetRealFlu(Queensland_Flu,p).values]
                     
df_real_flu = pd.DataFrame(real_flu,columns=Places,index = index)
'''    
    




TimeRange = [pd.datetime(2015,2,23).date(),pd.datetime(2015,8,10).date()]

#where_to_where_ =  GetPartOfTimeSeries(df_where_to_where,TimeRange)
#actual_days_in_week_  = GetPartOfTimeSeries(actual_days_in_week,TimeRange)
real_flu_ = GetPartOfTimeSeries(df_real_flu,TimeRange)[Places]

Times = real_flu_.index

# use each's move 


# (twitter_move_loc / twitter_all_loc * population_loc) * (real_flu_num_loc / population_loc)
# = (twitter_move_loc / twitter_all_loc  * real_flu_num_loc 

def norm(l):
    l = np.array(l)
    return (l-min(l))/(max(l)-min(l))     
     
     
plt.rc('figure',figsize = (16,20))


coef = []
for idx,place in enumerate(Places):
    
    twitter_move_loc_all = df_where_to_where.values[0][place][population_2015.keys()]*0
    for w2w in df_where_to_where.values:
        twitter_move_loc_all += w2w[place][population_2015.keys()]
    twitter_all_loc = df_twitter_in_place_loc.sum()
        
    
    y = real_flu_[place][1:].values
    x1 = real_flu_[place][:-1].values
    x2 = []
    for t in Times[:-1]:    
        #twitter_move_loc_t = df_where_to_where[t][place][population_2015.keys()]
        #twitter_all_loc_t = df_twitter_in_place_loc.loc[t]
        real_flu_num_loc_t = real_flu_.loc[t]
        #x2.append(sum(twitter_move_loc_t / twitter_all_loc_t * real_flu_num_loc_t))
        x2.append(sum(twitter_move_loc_all / twitter_all_loc * real_flu_num_loc_t))
    

    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model2 = LinearRegression()
        
    X = np.c_[x1,x2]
    
    model.fit(X,y)
    model2.fit(x1[:,np.newaxis],y)
    r1 = model.score(X,y)
    r2 = model2.score(x1[:,np.newaxis],y)
    
    
    
    y_pred = model.predict(X)
    y_pred_2 = model2.predict(x1[:,np.newaxis])
    

    
    plt.subplot(321+idx)
    
    y = pd.Series(y,index = Times[1:])
    y_pred = pd.Series(y_pred,index = Times[1:])
    x1 = pd.Series(y_pred_2,index = Times[1:])
    
    plt.plot(y)
    plt.plot(x1,'-.',alpha=0.8)
    plt.plot(y_pred,'-')

    plt.legend(['real','lag1_pred','pred'])
    plt.title(place+' | Population: '+str(population_2015[place]) + ' | R_1 : ' +str(round(r1,3)) + ', R_2: ' + str(round(r2,3)))
    
    coef.append(np.r_[model.coef_,model.intercept_])
    coef_dict = dict(zip(Places,coef))

    
    
# Model 

Pred = np.c_[real_flu_.loc[Times[0]].values]
for i in range(len(Times)-1):
    place_flu = []
    for place in Places:
        
        twitter_move_loc_all = df_where_to_where.values[0][place][population_2015.keys()]*0
        for w2w in df_where_to_where.values:
            twitter_move_loc_all += w2w[place][population_2015.keys()]
        twitter_all_loc = df_twitter_in_place_loc.sum()
            
        flu = pd.Series(Pred[:,i],index = population_2015.keys())
        
        x2 = sum(twitter_move_loc_all / twitter_all_loc * flu)
        x1 = flu[place]
        #y_1 = np.dot([x1,x2,1],coef_dict[place])
        y_1 = np.dot([x1,x2,1],np.array([1,6,0]))
        
        place_flu.append(y_1)
    Pred = np.c_[Pred,place_flu]
    
df_Pred = pd.DataFrame(Pred.T,columns=Places,index=Times)


real_flu_ = real_flu_[Places]
df_Pred = df_Pred[Places].applymap(lambda x: int(x))

Ratio = df_Pred.max()/real_flu_.max()

plt.figure()
plt.rc('figure',figsize = (10,7))

'''
plt.plot(real_flu_)
plt.plot(df_Pred/Ratio,'-.')
plt.legend(Places+Places , loc = 'upper left')
'''
plt.plot(real_flu_/real_flu_.max())
plt.plot(df_Pred/df_Pred.max(),'-.')
plt.legend(Places+Places , loc = 'upper left')



