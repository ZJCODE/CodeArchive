# -*- coding: utf-8 -*-
import numpy as np
import time
from datetime import datetime
from collections import Counter
import tensorflow as tf
import glob
from tensorflow.contrib.layers import fully_connected,layer_norm
import os


def load_test_data(test_data_path):
    test = open(test_data_path,'r')
    head = test.readline().strip().split(',')
    test_data = []
    for line in test.readlines():
        orderid,userid,bikeid,biketype,start_time,loc_start = line.strip().split(',')
        start_time = start_time.split('.')[0]
        dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        week = dt.weekday()
        test_data.append([userid,hour,week,loc_start])
    return test_data

def deal_with_node_vec_line(line,normalize = False):
    line = line.strip()
    line_split = line.split(' ')
    node = line_split[0]
    vec  = np.array(map(float,line_split[1:]))
    if normalize:
        vec = (vec - vec.mean()) / vec.std()
    return node,vec.tolist()

def load_node_vec_dict(node_vec_path,skip_line = False,normalize=False):
    f = open(node_vec_path,'r')
    node_vec_dict = {}
    if skip_line:
        f.readline()
    for line in f.readlines():
        node,vec = deal_with_node_vec_line(line,normalize)
        node_vec_dict[node] = vec
    return node_vec_dict

def deal_with_user_vec_line(line,normalize = False):
    line = line.strip()
    line_split = line.split(' ')
    user = line_split[0]
    vec  = np.array(map(float,line_split[1:]))
    if normalize:
        vec = (vec - vec.mean()) / vec.std()
    return user,vec.tolist()

def load_user_vec_dict(user_vec_path,skip_line = False,normalize=False):
    f = open(user_vec_path,'r')
    user_vec_dict = {}
    if skip_line:
        f.readline()
    for line in f.readlines():
        user,vec = deal_with_user_vec_line(line,normalize)
        user_vec_dict[user] = vec
    return user_vec_dict

def get_user_vec_dicts(user_vec_files_path):
    user_vec_dicts={}
    for path in user_vec_files_path:
        print 'process %s'%(path)
        kind = path.split('_vec_')[1]
        user_vec_dicts[kind] = load_user_vec_dict(path)
    return user_vec_dicts

def get_graph_user_vec_dicts(graph_user_files_path):
    graph_user_vec_dicts = {}
    for path in graph_user_files_path:
        print 'process %s'%(path)
        kind = path.split('_')[3]
        graph_user_vec_dicts[kind] = load_user_vec_dict(path)
    return graph_user_vec_dicts


node_vec_files = glob.glob('../data/loc_node_vec_order_combine*')
zip_path = zip(range(len(node_vec_files)),node_vec_files)
print 'we have those node vec files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
node_vec_path = node_vec_files[num]
test_data_path = '../data/test.csv'

loc_vec_size = int(node_vec_path.split('_dim_')[1])
user_vec_size = loc_vec_size *2 

user_vec_files_path = glob.glob('../data/user_vec*')
graph_user_files_path =  glob.glob('../data/vec_graph_user_*')
test_data = load_test_data(test_data_path)
node_vec_dict = load_node_vec_dict(node_vec_path)
node_vec_default = np.array(node_vec_dict.values()).mean(0)
user_vec_dicts = get_user_vec_dicts(user_vec_files_path)
user_vec_default = np.array(user_vec_dicts['all'].values()).mean(0)
graph_user_vec_dicts = get_graph_user_vec_dicts(graph_user_files_path)
graph_user_vec_default = np.array(graph_user_vec_dicts['whole'].values()).mean(0)


def get_test_input(test_data_line,node_vec_dict,user_vec_dicts,graph_user_vec_dicts):

    userid,hour,week,loc_start = test_data_line
    hour = int(hour)
    week = int(week)
    if hour >= 1 and hour <= 8 and week < 5:
        kind = 'hour_1_8_workday'
    elif hour >= 1 and hour <= 8 and week >= 5:
        kind = 'hour_1_8_weekday'
    elif hour >= 9 and hour <= 16 and week < 5:
        kind = 'hour_9_16_workday'
    elif hour >= 9 and hour <= 16 and week >= 5:
        kind = 'hour_9_16_weekday'
    elif (hour >= 17 or hour <= 0) and week < 5:
        kind = 'hour_17_24_workday'
    elif (hour >= 17 or hour <= 0) and week >= 5:
        kind = 'hour_17_24_weekday'

    
    try:
        graph_user_whole_vec = graph_user_vec_dicts['whole'][userid]
    except:
        graph_user_whole_vec = graph_user_vec_default

    try:
        graph_user_start_vec = graph_user_vec_dicts['start'][userid]
    except:
        graph_user_start_vec = graph_user_vec_default

    try:
        graph_user_end_vec = graph_user_vec_dicts['end'][userid]
    except:
        graph_user_end_vec = graph_user_vec_default

    try:
        user_vec_all = user_vec_dicts['all'][userid]
    except:
        #print 'did not see this user before'
        user_vec_all = user_vec_default

    try:
        user_vec_kind = user_vec_dicts[kind][userid]
    except:
        user_vec_kind =user_vec_default

    try:
        loc_start_vec = node_vec_dict[loc_start]
    except:
        print 'did not see this loc before'
        loc_start_vec = node_vec_default

    return [graph_user_whole_vec,graph_user_start_vec,graph_user_end_vec,user_vec_all,user_vec_kind,loc_start_vec,[hour],[week]]



saved_model_dir_all = glob.glob('../model/*')
zip_path = zip(range(len(saved_model_dir_all)),saved_model_dir_all)
print 'we have those model files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
saved_model_dir = saved_model_dir_all[num]


def model_inference(test_data):


    predcit_loc_end_path = '../data/predcit_loc_end_' + saved_model_dir.split('model_')[1]
    f = open(predcit_loc_end_path,'w')
    with tf.Session() as sess:
        
        signature_key = 'model_signature'
        graph_user_whole_vec_key = 'graph_user_whole_vec'
        graph_user_start_vec_key = 'graph_user_start_vec'
        graph_user_end_vec_key = 'graph_user_end_vec'
        user_vec_all_key = 'user_vec_all'
        user_vec_kind = 'user_vec_kind'
        loc_start_vec = 'loc_start_vec'
        hour_key = 'hour'
        week_key = 'week'
        keep_prob_key = 'keep_prob'
        predict_key = 'predict'

        meta_graph_def = tf.saved_model.loader.load(sess, ['model'], saved_model_dir)
        signature = meta_graph_def.signature_def

        graph_user_whole_vec_name = signature[signature_key].inputs[graph_user_whole_vec_key].name
        graph_user_start_vec_name = signature[signature_key].inputs[graph_user_start_vec_key].name
        graph_user_end_vec_name = signature[signature_key].inputs[graph_user_end_vec_key].name
        user_vec_all_name = signature[signature_key].inputs[user_vec_all_key].name
        user_vec_kind_name = signature[signature_key].inputs[user_vec_kind].name
        loc_start_vec_name = signature[signature_key].inputs[loc_start_vec].name
        hour_key_name = signature[signature_key].inputs[hour_key].name
        week_key_name = signature[signature_key].inputs[week_key].name
        keep_prob_name = signature[signature_key].inputs[keep_prob_key].name

        predict_name = signature[signature_key].outputs[predict_key].name

        graph_user_whole_vec = sess.graph.get_tensor_by_name(graph_user_whole_vec_name)
        graph_user_start_vec= sess.graph.get_tensor_by_name(graph_user_start_vec_name)
        graph_user_end_vec= sess.graph.get_tensor_by_name(graph_user_end_vec_name)
        user_vec_all = sess.graph.get_tensor_by_name(user_vec_all_name)
        user_vec_kind = sess.graph.get_tensor_by_name(user_vec_kind_name)
        loc_start_vec = sess.graph.get_tensor_by_name(loc_start_vec_name)
        hour = sess.graph.get_tensor_by_name(hour_key_name)
        week = sess.graph.get_tensor_by_name(week_key_name)
        keep_prob = sess.graph.get_tensor_by_name(keep_prob_name)
        predict = sess.graph.get_tensor_by_name(predict_name)
        i = 0
        for test_data_line in test_data:
            i += 1
            if i%10000 == 0:
                print 'process %d'%(i)
            x = get_test_input(test_data_line,node_vec_dict,user_vec_dicts,graph_user_vec_dicts)
            predict_value = sess.run(predict, feed_dict={graph_user_whole_vec : np.array(x[0]).reshape(1,-1),
                                                         graph_user_start_vec : np.array(x[1]).reshape(1,-1),
                                                         graph_user_end_vec_name : np.array(x[2]).reshape(1,-1),
                                                         user_vec_all: np.array(x[3]).reshape(1,-1),
                                                         user_vec_kind: np.array(x[4]).reshape(1,-1),
                                                         loc_start_vec: np.array(x[5]).reshape(1,-1),
                                                         hour: np.array(x[6]),
                                                         week: np.array(x[7]),
                                                         keep_prob: 1
                                        })
            f.write('-'.join(map(str,test_data_line)) + '|to|' + ' '.join(map(str,[round(a,5) for a in predict_value.tolist()[0]]))+'\n')
        f.close()
        #print predict_value


def main():
    model_inference(test_data)

if __name__ == '__main__':
    main()





