#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:24:30 2016

@author: ZJun
"""


from twitter_function import Import_Obj,GenerateDate,hinton,GetPartOfTimeSeries


population_2015 = dict(zip(['Cairns','Townsville','Mackay','Sunshine Coast','Brisbane','Gold Coast'],
                           [147993,180333,85455,302122,2209453,624918]))


df_flu_related_twitter_in_place_loc = Import_Obj('./DF_Result/df_flu_related_twitter_in_place_loc')[population_2015.keys()]
df_twitter_in_place_loc = Import_Obj('./DF_Result/df_twitter_in_place_loc')[population_2015.keys()]
df_move_destination_in_place_loc = Import_Obj('./DF_Result/df_move_destination_in_place_loc')[population_2015.keys()]
df_flu_related_move_destination_in_place_loc = Import_Obj('./DF_Result/df_flu_related_move_destination_in_place_loc')[population_2015.keys()]
df_real_flu = Import_Obj('./DF_Result/df_real_flu')[population_2015.keys()]
df_where_to_where = Import_Obj('./DF_Result/df_where_to_where')

df_flu_where_to_where = Import_Obj('./DF_Result/df_flu_where_to_where')
actual_days_in_week = Import_Obj('./Data/actual_days_in_week')

#a = df_where_to_where[GenerateDate(2015,1,12)]
#a.loc[population_2015.keys(),population_2015.keys()]

twitter_in_place_loc_sum = df_twitter_in_place_loc.sum()
place = twitter_in_place_loc_sum.index
real_twitter_ratio = []
for p in place:
    real_twitter_ratio.append(twitter_in_place_loc_sum[p]*1.0 / population_2015[p])
    
real_twitter_map_dict = dict(zip(place,real_twitter_ratio))    


TimeRange = [pd.datetime(2015,1,12).date(),pd.datetime(2015,3,16).date()]


'''
(df_real_flu['Brisbane'] * real_twitter_map_dict['Brisbane']).plot()
(df_flu_related_twitter_in_place_loc['Brisbane'] / actual_days_in_week*7).plot()
(df_flu_related_move_destination_in_place_loc['Brisbane']/actual_days_in_week*7).plot()
(df_move_destination_in_place_loc['Brisbane']*0.1/actual_days_in_week).plot()
(df_twitter_in_place_loc['Brisbane']*0.01/actual_days_in_week).plot()
'''
place = 'Brisbane'
(df_real_flu[place] * real_twitter_map_dict[place]).plot()
(df_move_destination_in_place_loc[place]/actual_days_in_week*0.1).plot()

# calculate how many flu people enter in to where 

