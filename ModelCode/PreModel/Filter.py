# -*- coding: utf-8 -*-
import numpy as np
import copy
import time
import json
from scipy import stats
import sys


def short_term_cluster_size(X,threshold):
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


def get_tj_feats(tj_data,loan_dt):
    tj_cnt = 0
    tj_tm_list = []
    tj_user_list = []
    tj_resource_list = []
    tj_query_price = []
    for item in tj_data:
        tj_cnt += 1
        tj_tm_list.append(int(item[tj_seq['create_tm']]))
        tj_user_list.append(int(item[tj_seq['user_id']]))
        tj_resource_list.append(int(item[tj_seq['resource_id']]))
        tj_query_price.append(float(item[tj_seq['query_price']]))
    if tj_cnt == 0:
        return ['']*10,0
    tj_tm_gap_list = [loan_dt-item for item in tj_tm_list]
    f1 = [tj_cnt]
    f2 = [len(set(tj_user_list))]
    f3 = [len(set(tj_resource_list))]
    time_window_cnt = parse_to_window(tj_tm_gap_list,day_to_secd([3,7,15,30,60,90]),type='cnt',cln_late_tm=365*86400,gap_parse=True)
    f4 = [round(np.std(time_window_cnt),4)]
    f5 = [len(filter(lambda x: x > 0, tj_query_price))]
    f6 = len(filter(lambda x: x <= 15*86400, tj_tm_gap_list)) #最近15天申请次数
    f7 = len(filter(lambda x: 15*86400 < x <= 30*86400, tj_tm_gap_list)) #最近16-30天申请次数
    f8 = len(filter(lambda x: 30*86400 < x <= 90*86400, tj_tm_gap_list)) #最近31-90天申请次数
    f9 = len(filter(lambda x: 90*86400 < x, tj_tm_gap_list)) #90天之外申请次数
    f10 = f6+f7+f8 #最近三个月申请次数


    return f1+f2+f3+f4+f5+[f6]+[f7]+[f8]+[f9]+[f10],1


def filter_data(data_list,tm_index=3):
    res = []
    sxjr_day = set()
    tjy_day = set()
    snjf_day = set()
    yzd_day = set()
    yzjr_day = set()
    krd_day = set()
    hzed_day = set()
    for item in data_list:
        day = int(item[tm_index])/86400
        # deal with 水象金融
        if int(item[tj_seq['user_id']]) == 326 :
            if day not in sxjr_day:
                sxjr_day.add(day)
                res.append(item)
        # deal with  融360淘金云项目
        elif int(item[tj_seq['user_id']]) == 94 and int(item[tj_seq['resource_id']]) == 282 :
            if day not in tjy_day:
                tjy_day.add(day)
                res.append(item)
        # 苏宁金服
        elif int(item[tj_seq['user_id']]) == 722 and int(item[tj_seq['resource_id']]) == 255 :
            if day not in snjf_day:
                snjf_day.add(day)
                res.append(item)
        # 融360原子贷
        elif int(item[tj_seq['user_id']]) == 174 and int(item[tj_seq['resource_id']]) == 215 :
            if day not in yzd_day:
                yzd_day.add(day)
                res.append(item)
        # 用钱宝
        elif int(item[tj_seq['user_id']]) == 673 and int(item[tj_seq['resource_id']]) == 217:
            pass
        # 意真金融
        elif int(item[tj_seq['user_id']]) == 221 and int(item[tj_seq['resource_id']]) in (217,294,305) :
            if day not in yzjr_day:
                yzjr_day.add(day)
                res.append(item)
        # 快人贷
        elif int(item[tj_seq['user_id']]) == 136 and int(item[tj_seq['resource_id']]) in (112,115,116,125) :
            if day not in krd_day:
                krd_day.add(day)
                res.append(item)
        # 合众E贷
        elif int(item[tj_seq['user_id']]) == 907 and int(item[tj_seq['resource_id']]) == 299 :
            if day not in hzed_day:
                hzed_day.add(day)
                res.append(item)
        # filter 内部机构
        elif int(item[tj_seq['user_id']]) in (572, 805, 1056, 328, 301, 94, 174, 355):
            pass
        else:
            res.append(item)
    return res



def trac_back_time(data_list,loan_dt,tm_index,logger=None):
    res = []
    for item in data_list:
        try:
            create_tm = int(item[tm_index])
        except:
            if logger:
                logger.info('create_tm is not int, is %s' % create_tm)
            create_tm = time.time()
        if create_tm < loan_dt and item not in res:
            res.append(item)
    return res

def tm_stamp_to_hour(tm_list):
    return [(i % 86400 / 3600 + 8 ) % 24 for i in tm_list]

def day_to_secd(ll):
    return [i*86400 for i in ll]

def parse_to_window(tm_gap_list, days_windows, type='cnt',cln_late_tm=False, gap_parse = True):
    '''tm_gap_list: 被分段的list； days_windows 统计区间；cln_late_tm  从tm_gap_list去掉大于的值； gap_parse 隔断&累积统计'''
    if cln_late_tm:
        tm_gap_list = [item for item in tm_gap_list if item < cln_late_tm]
    cnt_list = []
    for point in days_windows:
        cnt_list.append(len(filter(lambda x : x <= point, tm_gap_list)))
    cnt_diff_list = copy.deepcopy(cnt_list)
    for i in xrange(1,len(cnt_list)):
        cnt_diff_list[i] = cnt_list[i] - cnt_list[i-1]
    if gap_parse == True:
        cnt_list = cnt_diff_list
    if type=='cnt':
        return cnt_list
    elif type == 'pct':
        tol_cnt = len(tm_gap_list)
        tol_cnt = 1 if tol_cnt ==0 else tol_cnt
        return [float(i) / tol_cnt for i in cnt_list]



tj_seq ={'user_id':0,'tj_user_tag':1,'resource_id':2,'create_tm':3,'query_price':4}

filter_or_not = raw_input('Filter ? [y/n] : ')
if filter_or_not == 'y':
    output = open('../Data/tj_feature_label_with_filter','w')
else:
    output = open('../Data/tj_feature_label_without_filter','w')

with open('../Data/all_sample_0623_tj_raw_data','r') as f:
    for line in f.readlines():
        _,tj_data_raw,_,_,_,label,loan_dt,source = line.strip().split('\t')
        tj_data = json.loads(tj_data_raw)

        loan_dt = time.mktime(time.strptime(loan_dt,'%Y-%m-%d'))
        tj_data = trac_back_time(tj_data.get('tj',[]), loan_dt,tm_index=3, logger=None)

        if filter_or_not == 'y':
            tj_data = filter_data(tj_data,tm_index=3)
        if len(tj_data) > 0:
            tj_feature,feature_nan = get_tj_feats(tj_data,loan_dt)
            if feature_nan == 1:
                out_line = '\t'.join([str(x) for x in tj_feature]) + '\t'+ str(label) + '\t' + source + '\n'
                output.write(out_line)
output.close()
    