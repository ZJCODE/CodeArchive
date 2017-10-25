# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from pyvttbl import DataFrame as pDataFrame


def get_stat_from_tbl(tbl, label, filter_fields=['label','loan_dt','phone','op_id','mbl'], ks_threshold=0.07, result_dir='.'):
    df = pd.read_table(tbl, sep='\t')
    filter_fields.append(label)
    chi_file = result_dir + os.sep + tbl + '.chi.detail'
    ks_file = result_dir + os.sep + tbl + '.ks.detail'
    f_file = './' + tbl.split('/')[-1] + '.feature'
    
    feature_names = [feature for feature in df.keys() if feature not in filter_fields]
    out_names = 'label\tks\tiv\tcorr\tchi\tp\tbad_ratio\t' + '\t'.join(['count', 'mean', 'min', '25%', '50%', '75%', 'max']) + '\tbin_value'
    print out_names
    fks = open(ks_file, 'w')
    fchi = open(chi_file, 'w')
    fout = open(f_file, 'w')
    fout.write(out_names + '\n')

    for feature in feature_names:
        # ks, iv
        tmp_df = pd.DataFrame()
        tmp_df['good'] = 1 - df[label]
        tmp_df['bad'] = df[label]
        tmp_df['f'] = df[feature]
        #tmp_df = tmp_df[tmp_df['f'] > 0]
        #corr = tmp_df['bad'].corr(tmp_df['f'])
        group_info, result, ks, iv, bin_value = get_woe_iv_ks(df=tmp_df, bin_num=10)
        fks.write(feature + ':\n')
        fks.write('\n'.join(result) + '\n\n')
        output = feature + '\t' + str(ks) + '\t' + str(iv)
        # cov
        corr = df[label].corr(df[feature])
        output += ('\t' + str(corr))
        # chisq
        try:
            result = get_chi(df=df, feature_name=feature, label=label, range_list=['(-10000000,0]', '(0,100000000000]'])
        except Exception, e:
            output += '\t-1\t-1\t-1'
            result = str(e)
        else:
            output += ('\t' + str(result['chisq']) + '\t' + str(result['p']))+ '\t' + str(result.counter[(1,1)] / result.row_counter[1])
        fchi.write('-------------------------------------------------------------------------------------------\n')
        fchi.write(feature + ':\n')
        fchi.write(str(result) + '\n\n')
        # statistic
        stat_list = []
        #stat = df[feature][df[feature] > 0].describe()
        stat = df[feature].describe()
        for key in ['count', 'mean', 'min', '25%', '50%', '75%', 'max']:
            stat_list.append(str(stat[key]))
        output += ('\t' + '\t'.join(stat_list))

        # bin_value
        output += ('\t' + str(bin_value))

        if ks >= ks_threshold:
            fout.write(output + '\n')
        print output

    fks.close()
    fchi.close()
    fout.close()


def set_ks_group_func(bin_value, first_value=0):
    def func(value):
        for i in range(len(bin_value)):
            if value <= bin_value[i]:
                if i == 0:
                    return '[%s, %s]' % (first_value, bin_value[i])
                else:
                    return '(%s, %s]' % (bin_value[i-1], bin_value[i])
    return func


# 三列 f good bad
def get_woe_iv_ks(df, bin_num=5):
    f_uniq = copy.deepcopy(df['f'].drop_duplicates().get_values())
    f_uniq.sort()
    if len(f_uniq) == 0:
        return [], [], -1, -1, []
    if len(f_uniq) <= bin_num:
        df['group'] = df['f']
        bin_value = list(f_uniq)
        bin_value.sort()
    else:
        # 分段排序
        f_series = sorted(df['f'].get_values())
        f_cnt = len(f_series)
        bin_ratio = np.linspace(1.0/bin_num, 1, bin_num)
        bin_value = list(set([f_series[int(ratio*f_cnt)-1] for ratio in bin_ratio]))
        bin_value.sort()
        # 设置分组函数、添加分组信息、计数信息
        if f_series[0] < bin_value[0]:
            bin_value.insert(0, f_series[0])
        df['group'] = pd.cut(df['f'], bins=bin_value, precision=15, include_lowest=True)
        # 方式2，小数位更多
        #group_func = set_ks_group_func(bin_value, f_series[0])
        #df['group'] = map(group_func, df['f'])
    del df['f']
    group_info = df.groupby('group')
    group_sum_info = group_info.sum()
    group_sum_info['total'] = group_sum_info['good'] + group_sum_info['bad']
    total_good = sum(group_sum_info['good'])
    total_bad = sum(group_sum_info['bad'])
    total = total_good + total_bad
    group_sum_info['sample_ratio'] = group_sum_info['total'] / total

    group_sum_info['bad_ratio'] = group_sum_info['bad'] / total_bad
    group_sum_info['bad_ratio'] = group_sum_info['bad'] / total_bad
    group_sum_info['good_ratio'] = group_sum_info['good'] / total_good
    group_sum_info['woe'] = map(lambda x: 0 if x == 0 else math.log(x), group_sum_info['bad_ratio']/group_sum_info['good_ratio'])
    group_sum_info['iv'] = (group_sum_info['bad_ratio'] - group_sum_info['good_ratio']) * group_sum_info['woe']
    
    group_sum_info['cur_bad_ratio'] = group_sum_info['bad'] / group_sum_info['total']

    cumsum_info = group_sum_info.cumsum()
    group_sum_info['cum_good'] = cumsum_info['good']
    group_sum_info['cum_bad'] = cumsum_info['bad']
    group_sum_info['cum_good_ratio'] = group_sum_info['cum_good'] / total_good
    group_sum_info['cum_bad_ratio'] = group_sum_info['cum_bad'] / total_bad
    group_sum_info['ks'] = abs(group_sum_info['cum_good_ratio'] - group_sum_info['cum_bad_ratio'])

    ks = max(group_sum_info['ks'])
    iv = sum(group_sum_info['iv'])
    result = []
    result.append(u'区间\t总数\t坏样本\t坏占比\t好样本\t好占比\tWOE\tIV\t区间逾期率\t累积坏占比\t累积好占比\tKS')
    #print group_sum_info, group_sum_info['good'].keys()
    for row in range(len(group_sum_info)):
        out_list = [group_sum_info.index[row]]
        for key in ['total', 'bad', 'bad_ratio', 'good', 'good_ratio', 'woe', 'iv', 'cur_bad_ratio', 'cum_bad_ratio', 'cum_good_ratio', 'ks']:
            out_list.append(group_sum_info[key].iloc[row])
        result.append('\t'.join([str(item) for item in out_list]))
    return group_sum_info, result, ks, iv, bin_value


# range_list = ['(-1,0]', '(0,1]']
def get_chi(df, feature_name, label, range_list=None):
    start_time = datetime.now()
    pdf = pDataFrame()
    pdf[label] = df[label]
    pdf[feature_name] = df[feature_name]
    if range_list is not None:
        for i in range(len(pdf[feature_name])):
            for j in range(len(range_list)):
                item = range_list[j]
                low, high = item.strip()[1:-1].split(',')
                if pdf[feature_name][i] > int(low) and pdf[feature_name][i] <= int(high):
                    pdf[feature_name][i] = j
                    break
    result = pdf.chisquare2way(feature_name, label)
    return result


def set_ks_group_func(bin_value, first_value=0):
    def func(value):
        for i in range(len(bin_value)):
            if value <= bin_value[i]:
                if i == 0:
                    return '[%s, %s]' % (first_value, bin_value[i])
                else:
                    return '(%s, %s]' % (bin_value[i-1], bin_value[i])
    return func


def get_stat(tbl, label, filter_fields=['label','loan_dt','phone','op_id','mbl'], result_dir='.'):
    df = pd.read_table(tbl, sep='\t')
    filter_fields.append(label)
    chi_file = result_dir + os.sep + tbl + '.chi.detail'
    ks_file = result_dir + os.sep + tbl + '.ks.detail'
    
    feature_names = [feature for feature in df.keys() if feature not in filter_fields]
    out_names = '\t'.join(['label', 'ks', 'iv', 'corr', 'chi', 'bad_ratio']) + '\t' + \
            '\t'.join(['count', 'mean', 'min', '25%', '50%', '75%', 'max']) + '\t' + \
            '\t'.join(['ks2', 'iv2', 'corr2', 'count2', 'bin_value', 'bin_value2'])
    print out_names
    fks = open(ks_file, 'w')
    fchi = open(chi_file, 'w')

    for feature in feature_names:
        output = [feature]
        # ks, iv
        tmp_df = pd.DataFrame()
        tmp_df['good'] = 1 - df[label]
        tmp_df['bad'] = df[label]
        tmp_df['f'] = df[feature]
        tmp_df2 = copy.deepcopy(tmp_df[tmp_df['f'] > 0])
        # 新增逻辑
        corr2 = tmp_df2['bad'].corr(tmp_df2['f'])
        count2 = len(tmp_df2)
        # ks1 ks2
        group_info, result, ks, iv, bin_value = get_woe_iv_ks(df=tmp_df, bin_num=10)
        group_info2, result2, ks2, iv2, bin_value2 = get_woe_iv_ks(df=tmp_df2, bin_num=10)
        fks.write(feature + ':\n')
        fks.write('\n'.join(result) + '\n\n')
        fks.write('\n'.join(result2) + '\n\n')
        output.append(ks)
        output.append(iv)
        # corr
        corr = df[label].corr(df[feature])
        output.append(corr)
        # chisq
        try:
            result = get_chi(df=df, feature_name=feature, label=label, range_list=['(-10000000,0]', '(0,100000000000]'])
        except Exception, e:
            output.append(-1)
            output.append(-1)
            result = str(e)
        else:
            output.append(result['chisq'])
            output.append(result.counter[(1,1)] / result.row_counter[1])
        fchi.write(feature + ':\n')
        fchi.write(str(result) + '\n\n')
        # statistic
        stat_list = []
        #stat = df[feature][df[feature] > 0].describe()
        stat = df[feature].describe()
        for key in ['count', 'mean', 'min', '25%', '50%', '75%', 'max']:
            output.append(stat[key])
        # 新增信息
        output.extend([ks2, iv2, corr2, count2, bin_value, bin_value2])
        output = '\t'.join([str(item) for item in output])
        print output

    fks.close()
    fchi.close()


# 三列 f good bad
def get_woe_iv_ks_by_bin_value(df, bin_value, sample_ratio):
    df['group'] = pd.cut(df['f'], bins=bin_value, precision=15, include_lowest=True)
    del df['f']
    group_info = df.groupby('group')
    group_sum_info = group_info.sum()
    # 填补缺失值
    group_sum_info['good'].fillna(0)
    group_sum_info['bad'].fillna(0)
    group_sum_info['total'] = group_sum_info['good'] + group_sum_info['bad']
    total_good = sum(group_sum_info['good'])
    total_bad = sum(group_sum_info['bad'])
    total = total_good + total_bad
    group_sum_info['bad_ratio'] = group_sum_info['bad'] / total_bad
    group_sum_info['good_ratio'] = group_sum_info['good'] / total_good
    group_sum_info['woe'] = map(lambda x: 0 if x == 0 else math.log(x), group_sum_info['bad_ratio']/group_sum_info['good_ratio'])
    group_sum_info['iv'] = (group_sum_info['bad_ratio'] - group_sum_info['good_ratio']) * group_sum_info['woe']

    # psi
    group_sum_info['psi'] = (group_sum_info['total']/total - sample_ratio) * map(lambda x: 0 if x == 0 else math.log(x), (group_sum_info['total'] / total) / sample_ratio)
    # 段间逾期率
    group_sum_info['cur_bad_ratio'] = group_sum_info['bad'] / group_sum_info['total']

    cumsum_info = group_sum_info.cumsum()
    group_sum_info['cum_good'] = cumsum_info['good']
    group_sum_info['cum_bad'] = cumsum_info['bad']
    group_sum_info['cum_good_ratio'] = group_sum_info['cum_good'] / total_good
    group_sum_info['cum_bad_ratio'] = group_sum_info['cum_bad'] / total_bad
    group_sum_info['ks'] = abs(group_sum_info['cum_good_ratio'] - group_sum_info['cum_bad_ratio'])

    ks = max(group_sum_info['ks'])
    iv = sum(group_sum_info['iv'])
    psi = sum(group_sum_info['psi'])
    result = []
    result.append(u'区间\t总数\t坏样本\t坏占比\t好样本\t好占比\tWOE\tIV\t区间逾期率\t累积坏占比\t累积好占比\tKS\tPSI')
    #print group_sum_info, group_sum_info['good'].keys()
    for row in range(len(group_sum_info)):
        out_list = [group_sum_info.index[row]]
        for key in ['total', 'bad', 'bad_ratio', 'good', 'good_ratio', 'woe', 'iv', 'cur_bad_ratio', 'cum_bad_ratio', 'cum_good_ratio', 'ks', 'psi']:
            out_list.append(group_sum_info[key].iloc[row])
        result.append('\t'.join([str(item) for item in out_list]))
    return group_sum_info, result, ks, iv, psi, bin_value


def get_stat_compare(tbl, tbl2, label, filter_fields=['label','loan_dt','phone','op_id','mbl'], result_dir='.', prefix=''):
    df = pd.read_table(tbl, sep='\t')
    df2 = pd.read_table(tbl2, sep='\t')
    filter_fields.append(label)
    ks_file = result_dir + os.sep + tbl2 + '.ks.detail'
    
    feature_names = [feature for feature in df.keys() if feature not in filter_fields]
    out_names = '\t'.join(['label', 'ks', 'iv', 'psi', 'corr'])
    print out_names
    fks = open(ks_file, 'w')

    for feature in feature_names:
        output = [prefix+feature]
        # ks, iv
        tmp_df = pd.DataFrame()
        tmp_df['good'] = 1 - df[label]
        tmp_df['bad'] = df[label]
        tmp_df['f'] = df[feature]
        # ks
        group_info, result, ks, iv, bin_value = get_woe_iv_ks(df=tmp_df, bin_num=10)
        
        # 待验证样本
        tmp_df = pd.DataFrame()
        tmp_df['good'] = 1 - df2[label]
        tmp_df['bad'] = df2[label]
        tmp_df['f'] = df2[feature]
        #bin_value[-1] = 1.0
        sample_ratio = list(group_info['sample_ratio'])
        #print group_info, bin_value, '----------------'
        if len(bin_value) == 2:
            bin_value.insert(1, sum(bin_value)/2.0)

        try:
            group_info2, result2, ks2, iv2, psi2, bin_value2 = get_woe_iv_ks_by_bin_value(df=tmp_df, bin_value=bin_value, sample_ratio=sample_ratio)
        except:
            group_info2, result2, ks2, iv2, psi2, bin_value2 = None, ['error'], -1, -1, -1, None
        fks.write(feature + ':\n')
        fks.write('\n'.join(result2) + '\n\n')
        output.append(ks2)
        output.append(iv2)
        output.append(psi2)
        # corr
        corr = df2[label].corr(df2[feature])
        output.append(corr)
        output = '\t'.join([str(item) for item in output])
        print output

    fks.close()



if __name__ == '__main__':
    get_stat_compare(sys.argv[1], sys.argv[2], "0")
