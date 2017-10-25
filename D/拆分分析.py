#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:01:09 2017

@author: zhangjun
"""

import pandas as pd


city_list = ['北京','杭州','成都','哈尔滨','深圳']



city_data = pd.read_csv('市内.csv','\t')
cross_data = pd.read_csv('跨城.csv','\t')

city_road = pd.read_csv('市内热门路线.csv','\t')
cross_road = pd.read_csv('跨城热门路线.csv','\t')

city_data.columns = ['日期','城市','市内司乘比','市内日呼叫订单','市内应答率','市内成功送达率','市内平均订单价格','市内平均订单距离','市内应答前取消率']

cross_data.columns = ['日期','城市','跨城司乘比','跨城日呼叫订单','跨城应答率','跨城成功送达率','跨城平均订单价格','跨城应答前取消率']

city_data.to_csv('市内详情.csv',index=False,header=True,encoding='gbk')
cross_data.to_csv('跨城详情.csv',index=False,header=True,encoding='gbk')


cross_road['from_city'] = [x.split('-')[0] for x in cross_road.road]
cross_road['to_city'] = [x.split('-')[1] for x in cross_road.road]

cross_road_start_focus = cross_road[cross_road.from_city.isin(city_list)]
                              
cross_road_start_focus = cross_road_start_focus.sort_values('order_cnt',ascending=False)

cross_road_focus_top_10 = cross_road_start_focus.head(10)
cross_road_focus_top_10.columns = ['路线','近2月订单量','单均价','单均距离','出发地','目的地']
cross_road_focus_top_10.to_csv('跨城热门路线Top10.csv',index=False,header=True,encoding='gbk')

cross_road_start_focus = cross_road[cross_road.from_city.isin(city_list)]
cross_road_end_focus = cross_road[cross_road.to_city.isin(city_list)]
                                  
                      
cross_road_start_focus['all_distance'] = cross_road_start_focus['order_cnt']*cross_road_start_focus['navi_distance']
cross_road_end_focus['all_distance'] = cross_road_end_focus['order_cnt']*cross_road_end_focus['navi_distance']
cross_road_start_focus_pt = cross_road_start_focus.pivot_table(['all_distance','order_cnt'],'from_city',aggfunc='sum')
cross_road_end_focus_pt = cross_road_end_focus.pivot_table(['all_distance','order_cnt'],'to_city',aggfunc='sum')

cross_road_start_focus_avg_distance = cross_road_start_focus_pt.all_distance/cross_road_start_focus_pt.order_cnt
cross_road_end_focus_avg_distance = cross_road_end_focus_pt.all_distance/cross_road_end_focus_pt.order_cnt

cross_road_start_focus_avg_distance.to_csv('作为起点城市订单平均距离.csv',index=True,encoding='gbk')
cross_road_end_focus_avg_distance.to_csv('作为终点城市订单平均距离.csv',index=True,encoding='gbk')

for city in city_list:
    select_city_data = city_road[city_road.city_name == city]
    select_city_data = select_city_data.sort_values('order_cnt',ascending=False)
    select_city_data_top_100 = select_city_data.head(100)
    select_city_data_top_100.columns = ['城市','路线','近2月订单量','单均价']
    select_city_data_top_100.to_csv(city+'市内Top100路线.csv',index=False,header=True,encoding='gbk')
    
for city in city_list:
    select_city_data = city_road[city_road.city_name == city]
    select_city_data['from_name'] = [x.split('|->|')[0] for x in select_city_data.from_to_name]
    select_city_data['to_name'] = [x.split('|->|')[1] for x in select_city_data.from_to_name]
    temp1 = select_city_data[['from_name','order_cnt']]
    temp1.columns = ['name','cnt']
    temp2 = select_city_data[['to_name','order_cnt']]
    temp2.columns = ['name','cnt']
    df = pd.concat([temp1,temp2])
    top100_place = df.pivot_table('cnt','name',aggfunc='sum').sort_values(ascending=False)[:100]
    top100_place.to_csv(city+'Top100热门商圈及其近2个月出现次数.csv',encoding='gbk')