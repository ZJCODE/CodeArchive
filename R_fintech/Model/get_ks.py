# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
#import seaborn
#seaborn.set()
#import matplotlib.pyplot as plt

KS_PART = 10

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) / 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

def ks(y_true, y_prob, ks_part=KS_PART):
    
    nan_ratio = round(sum(np.isnan(y_prob))*1.0/len(y_prob) *100,1)
    zero_ratio = round((len(y_prob) - np.count_nonzero(y_prob))*1.0 / len(y_prob) *100 ,1)
    
    data = np.vstack((y_true, y_prob)).T
    #--Remove nan--
    data = data[~np.isnan(data).any(1)]
    #------
    sort_ind = np.argsort(data[:, 1])
    data = data[sort_ind]

    length = len(data)
    sum_bad = sum(data[:, 0])
    sum_good = length - sum_bad

    cut_list = [0]
    
    order_num = []
    bad_num = []

    cut_pos_last = -1
    for i in np.arange(ks_part):
        if i == ks_part-1 or data[length*(i+1)/ks_part-1, 1] != data[length*(i+2)/ks_part-1, 1]:
            cut_list.append(data[length*(i+1)/ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[length*(i+1)/ks_part-1, 1], data[:, 1], length*(i+1)/ks_part-1, length*(i+2)/ks_part-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(sum(data[cut_pos_last+1:cut_pos+1, 0]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)

    good_num = order_num - bad_num
    order_ratio = np.array([round(x, 3) for x in order_num * 100 / float(length)])
    overdue_ratio = np.array([round(x, 3) for x in bad_num * 100 / [float(x) for x in order_num]])
    bad_ratio = np.array([round(sum(bad_num[:i+1])*100/float(sum_bad), 3) for i in range(len(bad_num))])
    good_ratio = np.array([round(sum(good_num[:i+1])*100/float(sum_good), 3) for i in range(len(good_num))])
    ks_list = abs(good_ratio - bad_ratio)
    ks = max(ks_list)

    try:
        span_list = ['[%.2f,%.2f]' % (min(data[:, 1]), round(cut_list[1], 3))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                span_list.append('(%.2f,%.2f]' % (round(cut_list[i-1], 3), round(cut_list[i], 3)))
    except:
        span_list = ['0']

    dic_ks = {
            'ks': ks,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'order_ratio': order_ratio,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio,
            'good_ratio': good_ratio,
            'ks_list': ks_list,
            'except_ratio':{'nan_ratio':nan_ratio,'zero_ratio':zero_ratio}
            }

    return dic_ks

def print_ks(ks_info,f=''):
    line = ''
    print 'f=%s\tks = %.1f%%' % (f, ks_info['ks'])
    #line = line + 'f=%s\tks = %.1f%%' % (f, ks_info['ks']) + '\n'
    print '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量'])
    #line = line + '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量']) + '\n'
    for i in range(len(ks_info['ks_list'])):
        print '%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i])
        #line = line + '%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i]) + '\n'
    return line

def print_ks_(ks_info,f=''):
    line = ''
    print 'f=%s\tks = %.1f%%' % (f, ks_info['ks'])
    line = line + 'f=%s\tks = %.1f%%' % (f, ks_info['ks']) + '\n'
    print '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量'])
    line = line + '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量']) + '\n'
    for i in range(len(ks_info['ks_list'])):
        print '%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i])
        line = line + '%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i]) + '\n'
    return line

import os

def plot_ks(dict_ks,f='',place = 'Here'):
    span_list = dict_ks['span_list']
    order_ratio = dict_ks['order_ratio']
    order_num = dict_ks['order_num']
    overdue_ratio = dict_ks['overdue_ratio']
    good_ratio = dict_ks['good_ratio']
    bad_ratio = dict_ks['bad_ratio']
    nan_ratio = dict_ks['except_ratio']['nan_ratio']
    zero_ratio = dict_ks['except_ratio']['zero_ratio']
    pos = np.arange(len(span_list))
    plt.rc('figure',figsize=[18,20])
    plt.figure()
    plt.subplot(311)
    plt.bar(pos,order_ratio,alpha = 0.7,width=0.6)
    plt.ylim([min(order_ratio),max(order_ratio) + 5])
    for x,y,s in zip(pos,order_ratio,map(str,order_num)):        
        plt.text(x,y+0.5,s)
    plt.ylabel('Percent ( % )')
    plt.legend(['order_ratio'],fontsize=15)
    plt.xticks(pos,span_list)
    plt.title( 'Feature [ '+f+' ] Summary | ' + 'KS : %.3f |  Featur Nan : %.1f%% | Feature Zero : %.1f%%' %(dict_ks['ks'],nan_ratio,zero_ratio),fontsize=15)
    plt.subplot(312)
    plt.plot(pos,overdue_ratio,'mo-',alpha = 0.8)
    plt.ylabel('Percent ( % )')
    plt.legend(['overdue_ratio'],fontsize=15)
    plt.xticks(pos,span_list)
    plt.subplot(313)
    plt.plot(pos,good_ratio,'r',alpha = 0.8)
    plt.plot(pos,bad_ratio,'g',alpha = 0.8)
    #plt.ylim([0,100])
    plt.legend(['good_ratio','bad_ratio','ks'],fontsize=15)
    plt.xticks(pos,span_list)
    #plt.show()
    file_path = './Pic'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path = './Pic/' +place
    if not os.path.exists(file_path):
        os.mkdir(file_path)  
    path = file_path +'/KS_'+ str(round(dict_ks['ks'],3))+'_'+ 'Nan_' + str(nan_ratio) + '_Zero_' + str(zero_ratio) + '_'+f+'.png'
    plt.savefig(path)
    plt.close()
    
    
    
