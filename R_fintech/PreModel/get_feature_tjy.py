# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 17:20:41 2017

@author: zhangjun
"""

import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from collections import Counter

order_columns_dict = dict(zip(['mbl_num','tjy_order_id','city_name','bank_name','product_name','application_amount',
            'application_term','application_term_unit','order_stat','approve_amt',
            'conf_loan_amt','order_create_tm','last_update_tm','is_inner_sett','label','loan_dt','source'],range(20)))

def day2tm(x):
    x = int(x)
    return 86400*x
def between(x,a,b):
    return (x>=a) & (x<=b)

# 周期性，规律性
# 产品维度
# 交互产品计数

def continue_act(X,threshold):
    X_diff = np.diff(sorted(X))
    max_size = 0 
    size = 0
    for x in X_diff:
        if x < threshold:
            size += 1
        else:
            size = 0
        max_size = max(size,max_size)
    return max_size
    
def statistics(X):
    if len(X)>0:        
        X = np.array(X)
        sum_ = np.sum(X)
        mean_ = np.mean(X)
        std_ = np.std(X)
        max_ = np.max(X)
        return [sum_,mean_,std_,max_]
    else:
        return [0,0,0,0]
    
def more_than_ratio(X,threshold):
    X = np.array(X)
    return len(X[X>threshold])*1.0/len(X)

def no_more_than_ratio(X,threshold):
    X = np.array(X)
    return len(X[X<=threshold])*1.0/len(X)

def get_order_feature(line):
    
    line_split = line.rstrip().split('\t')
    label,loan_dt,source =  line_split[-3:]
    loan_dt_tm = time.mktime(time.strptime(loan_dt,'%Y-%m-%d'))
    tjy_order_id = np.array(line_split[order_columns_dict['tjy_order_id']].split(','))
    order_create_tm = np.array(map(int,line_split[order_columns_dict['order_create_tm']].split(',')))
    
    
    if len(tjy_order_id[order_create_tm < loan_dt_tm]) > 0:
        tjy_order_id = np.array(line_split[order_columns_dict['tjy_order_id']].split(','))[order_create_tm < loan_dt_tm]
        application_amount = np.array(map(float,line_split[order_columns_dict['application_amount']].split(',')))[order_create_tm < loan_dt_tm]
        conf_loan_amt = np.array(map(float,line_split[order_columns_dict['conf_loan_amt']].split(',')))[order_create_tm < loan_dt_tm]
        bank_name = np.array(line_split[order_columns_dict['bank_name']].split(','))[order_create_tm < loan_dt_tm]
        product_name = np.array(line_split[order_columns_dict['product_name']].split(','))[order_create_tm < loan_dt_tm]
        application_term = np.array(map(int,line_split[order_columns_dict['application_term']].split(',')))[order_create_tm < loan_dt_tm]
        application_term_tm = np.array(map(day2tm,line_split[order_columns_dict['application_term']].split(',')))[order_create_tm < loan_dt_tm]
        order_create_tm = np.array(map(int,line_split[order_columns_dict['order_create_tm']].split(',')))[order_create_tm < loan_dt_tm]
        
        plan_repay_tm = order_create_tm + application_term_tm
        conf_order_create_tm = order_create_tm[conf_loan_amt>0]
        diff_plan_loan_dt = (plan_repay_tm[conf_loan_amt>0] - loan_dt_tm)/86400
        application_conf_amt_diff = application_amount - conf_loan_amt
    
        #-------------
        f0 = len(tjy_order_id)
        # bank_cnt 
        f = len(set(bank_name))
        f1 = [f,f*1.0/f0]
        # product_cnt 
        f = len(set(product_name))
        f2 = [f,f*1.0/f0]
        # bank_cnt_1M 
        f = len(set([x[0] for x in zip(bank_name,order_create_tm) if between(x[1] - loan_dt_tm,-30*86400,0)]))
        f3 = [f,f*1.0/f0]
        # product_cnt_1M 
        f = len(set([x[0] for x in zip(product_name,order_create_tm) if between(int(x[1]) - loan_dt_tm,-30*86400,0)]))
        f4 = [f,f*1.0/f0]
        #-------------
        # in_progress_order_cnt 
        f = sum((conf_loan_amt >0) & (plan_repay_tm > loan_dt_tm))
        f5 = [f,f*1.0/f0]
        #-------------
        # continue_act_apply_1 
        f6 = continue_act(order_create_tm/86400,1)
        # continue_act_apply_2 
        f7 = continue_act(order_create_tm/86400,2)
        # continue_act_5 
        f8 = continue_act(order_create_tm/86400,5)
        # continue_act_10 
        f9 = continue_act(order_create_tm/86400,10)
        #-------------
        # continue_act_15 
        f10 = continue_act(conf_order_create_tm/86400,15)
        # continue_act_30 
        f11 = continue_act(conf_order_create_tm/86400,30)
        #-----------
        f12 = statistics(application_amount)
        f13 = statistics(conf_loan_amt)
        f14 = statistics(conf_loan_amt[conf_loan_amt>0])
        f15 = statistics(application_conf_amt_diff)
        f16 = statistics(application_term)
        f17 = statistics(diff_plan_loan_dt)
        f18 = statistics(np.diff(sorted(conf_order_create_tm/86400)))
        #-----------
        f19 = [more_than_ratio(application_amount,3000),more_than_ratio(application_amount,6000)]
        f20 = [more_than_ratio(conf_loan_amt,0),more_than_ratio(conf_loan_amt,3000),more_than_ratio(conf_loan_amt,6000)]
        f21 = [more_than_ratio(application_term,60),more_than_ratio(conf_loan_amt,180)]
        #------------------------
        f22 = [len(set(bank_name[conf_loan_amt>0])),len(set(bank_name[conf_loan_amt>0]))*1.0/len(set(bank_name))]
        f23 = [len(set(product_name[conf_loan_amt>0])),len(set(product_name[conf_loan_amt>0]))*1.0/len(set(product_name))]
    #------------
        feature = [f0]+f1+f2+f3+f4+f5+[f6]+[f7]+[f8]+[f9]+[f10]+[f11]+f12+f13+f14+f15+f16+f17+f18+f19+f20+f21+f22+f23
    else:
        feature = [np.nan]*56
    
    return line_split[0],loan_dt,feature,label,source
    

def get_feature_name():
    feature_name = []
    feature_name += ['order_cnt']
    feature_name += ['bank'+ x for x in ['_cnt','_ratio']]
    feature_name += ['product'+ x for x in ['_cnt','_ratio']]
    feature_name += ['bank_1M'+ x for x in ['_cnt','_ratio']]
    feature_name += ['product_1M'+ x for x in ['_cnt','_ratio']]
    feature_name += ['in_progress'+ x for x in ['_cnt','_ratio']]
    feature_name += ['continue_act_apply_' + str(x) for x in [1,2,5,10]]
    feature_name += ['continue_act_conf_' + str(x) for x in [15,30]]
    feature_name += ['application_amount' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['conf_loan_amt' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['conf_loan_amt_more_than_0' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['application_conf_amt_diff' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['application_term' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['diff_plan_loan_dt' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['diff_conf_order_create_tm' + x for x in ['_sum','_mean','_std','_max']]
    feature_name += ['application_amount_more_than_' + str(x) + '_ratio' for x in [3000,6000]]
    feature_name += ['conf_loan_amt_more_than_' + str(x) + '_ratio' for x in [0,3000,6000]]
    feature_name += ['application_term_more_than_' + str(x) + '_ratio' for x in [60,180]]
    feature_name += ['conf_loan_bank_' + x for x in ['_cnt','_ratio']]
    feature_name += ['conf_loan_product_' + x for x in ['_cnt','_ratio']]
    feature_name = ['tjy_' + x for x in feature_name]
    return feature_name
    
output = open('../Data/tjy_feature','w')
intput = open('../Data/tjy_order_combine_with_label','r')
output_featrue_name = open('../Data/tjy_feature_name','w')
line_num = 0
lines = intput.readlines()
t1 = time.time()
for line in lines:
    if line_num %2000 == 0:
        print line_num
        t2 = time.time()
        print 'cost : %f min'%((t2-t1)/60)
        t1 = t2
    line_num +=1
    mbl,loan_dt,feature,label,source = get_order_feature(line)
    output.write(mbl+'\t'+loan_dt+'\t'+label+'\t'+source+'\t')
    output.write('\t'.join([str(i) for i in feature]))
    output.write('\n')
output.close()
feature_name = get_feature_name()
output_featrue_name.write('\t'.join(feature_name))
output_featrue_name.close()




#------------------------------------------------------------------------------

log_columns_dict = dict(zip(['mbl_num','stat','tjy_order_id','log_create_dt','label','loan_dt','source'],range(7)))

# 提前多少天还款等信息挖掘
# 主动放弃借款
# 逾期时间长度

def get_order_log_feature(line):
        line_split = line.rstrip().split('\t')
        label,loan_dt,source =  line_split[-3:]
        #tjy_order_id = np.array(line_split[log_columns_dict['tjy_order_id']].split(','))
        log_create_dt = np.array(line_split[log_columns_dict['log_create_dt']].split(','))
        stat = np.array(line_split[log_columns_dict['stat']].split(','))[log_create_dt < loan_dt].tolist()
        tjy_order_id = np.array(line_split[log_columns_dict['tjy_order_id']].split(','))[log_create_dt < loan_dt].tolist()
        log_create_dt = np.array(line_split[log_columns_dict['log_create_dt']].split(','))[log_create_dt < loan_dt].tolist()
        log_create_dt = [time.mktime(time.strptime(x,'%Y-%m-%d')) for x in log_create_dt]
        
        if len(stat) >0:
            
            f1 = stat.count('40') + stat.count('42') + stat.count('88') + stat.count('110') \
                +stat.count('135') +stat.count('152')+stat.count('169')+stat.count('135')
                
            f2 = stat.count('180') + stat.count('181')+stat.count('186')
            
            f3 = f1+f2
            
            f4 = stat.count('86')+ stat.count('100') + stat.count('115') + stat.count('162') \
                + stat.count('170')
                
            f5 = stat.count('190') +stat.count('195')+ stat.count('175')
            
            f6 = f4 + f5
            # user_giveup
            f7 = stat.count('50')
            
            # order_duration
            id_tm_zip = sorted(zip(tjy_order_id,log_create_dt),key=lambda x:x[0])
            order_id_pre , log_tm = id_tm_zip[0]
            t_min = log_tm
            t_max = log_tm
            order_duration = []
            for order_id , log_tm in id_tm_zip:
                if order_id == order_id_pre:
                    t_min = min(t_min,log_tm)
                    t_max = max(t_max,log_tm)
                else:
                    order_duration.append((t_max-t_min)/86400)
                    t_min = log_tm
                    t_max = log_tm
                order_id_pre = order_id
            order_duration.append((t_max-t_min)/86400)
            
            f8 = statistics(order_duration)
            f9 = [no_more_than_ratio(order_duration,10),no_more_than_ratio(order_duration,15),\
                  no_more_than_ratio(order_duration,30),no_more_than_ratio(order_duration,60)]
            
            # repay time 
            order_stat_tm_zip = sorted(zip(tjy_order_id,stat,log_create_dt),key=lambda x:x[2])
            repay_time_list = []
            order_id_pre,_,_ = order_stat_tm_zip[0]
            log_tm_start = -1
            log_tm_end = -1
            for order_id,st,log_tm in order_stat_tm_zip:
                if order_id == order_id_pre:
                    if st == '170':
                        log_tm_start = log_tm
                    if st == '190' or st =='200':
                        log_tm_end = log_tm
                elif order_id != order_id_pre and log_tm_start>0 and log_tm_end == -1:
                    repay_time_list.append(10000)
                    log_tm_start = -1
                    log_tm_end = -1
                elif order_id != order_id_pre and log_tm_start>0 and log_tm_end > 0:
                    repay_time_list.append((log_tm_end -log_tm_start)/86400)
                    log_tm_start = -1
                    log_tm_end = -1
                order_id_pre = order_id
            if log_tm_start > 0 and log_tm_end == -1:
                repay_time_list.append(10000)
            elif log_tm_start>0 and log_tm_end>0:
                repay_time_list.append((log_tm_end -log_tm_start)/86400)
            
            if len(repay_time_list)>0:
                f10 = [no_more_than_ratio(repay_time_list,10),no_more_than_ratio(repay_time_list,15),\
                      no_more_than_ratio(repay_time_list,30),no_more_than_ratio(repay_time_list,60)]
            else:
                f10 = [np.nan]*4
                
            feature = [f1]+[f2]+[f3]+[f4]+[f5]+[f6]+[f7] + f8 + f9 + f10
        else:
            feature = [np.nan] * 19
        
        return line_split[0],feature,label,loan_dt,source
    
def get_order_log_feature_name():
    feature_name = []
    feature_name += ['n1','n2','n3']
    feature_name += ['p1','p2','p3']
    feature_name += ['user_giveup']
    feature_name += ['order_duration' + x for x in ['_sum','_mean','_std','_max'] ]
    feature_name += ['order_duration_no_more_than_' + str(x) for x in [10,15,30,60]  ]
    feature_name += ['repay_time_no_more_than_' + str(x) for x in [10,15,30,60]  ]
    feature_name = ['tjy_log_' + x for x in feature_name]
    return feature_name


output = open('../Data/tjy_log_feature','w')
intput = open('../Data/stat_log_combine_with_label','r')
output_featrue_name = open('../Data/tjy_log_feature_name','w')
line_num = 0
lines = intput.readlines()
t1 = time.time()
for line in lines:
    if line_num %2000 == 0:
        print line_num
        t2 = time.time()
        print 'cost : %f min'%((t2-t1)/60)
        t1 = t2
    line_num +=1
    mbl,feature,label,loan_dt,source = get_order_log_feature(line)
    output.write(mbl+'\t'+loan_dt+'\t'+label+'\t'+source+'\t')
    output.write('\t'.join([str(i) for i in feature]))
    output.write('\n')
output.close()
feature_name = get_order_log_feature_name()
output_featrue_name.write('\t'.join(feature_name))
output_featrue_name.close()
        