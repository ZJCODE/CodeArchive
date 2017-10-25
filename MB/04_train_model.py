# -*- coding: utf-8 -*-
import numpy as np
import time
from datetime import datetime
from collections import Counter
import tensorflow as tf
import glob
from tensorflow.contrib.layers import fully_connected,layer_norm
import os




def load_train_data(tarin_data_path):
    train = open(tarin_data_path,'r')
    head = train.readline().strip().split(',')
    tarin_data = []
    for line in train.readlines():
        orderid,userid,bikeid,biketype,start_time,loc_start,loc_end = line.strip().split(',')
        dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        week = dt.weekday()
        tarin_data.append([userid,hour,week,loc_start,loc_end])
    return tarin_data


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


def generate_sample(train_data,node_vec_dict,user_vec_dicts,graph_user_vec_dicts,shuffle = False):
    while True:
        if shuffle:
            np.random.shuffle(train_data)    
        for line in train_data:
            userid,hour,week,loc_start,loc_end = line
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
            yield graph_user_vec_dicts['whole'][userid],graph_user_vec_dicts['start'][userid],\
                  graph_user_vec_dicts['end'][userid],user_vec_dicts['all'][userid],user_vec_dicts[kind][userid],\
                  node_vec_dict[loc_start],node_vec_dict[loc_end],hour,week


def get_batch(iterator,batch_size):
    while True:
        graph_user_whole_batch = []
        graph_user_start_batch = []
        graph_user_end_batch = []
        user_vec_all_batch = []
        user_vec_kind_batch = []
        loc_start_vev_batch = []
        loc_end_vec_batch = []
        hour_batch = []
        week_batch = []
        for i in range(batch_size):
            g_u_w,g_u_s,g_u_e,u_v_a , u_v_k , l_s_v ,l_e_v ,h ,w = next(iterator)
            graph_user_whole_batch.append(g_u_w)
            graph_user_start_batch.append(g_u_s)
            graph_user_end_batch.append(g_u_e)
            user_vec_all_batch.append(u_v_a)
            user_vec_kind_batch.append(u_v_k)
            loc_start_vev_batch.append(l_s_v)
            loc_end_vec_batch.append(l_e_v)
            hour_batch.append(h)
            week_batch.append(w)
        graph_user_whole_batch = np.array(graph_user_whole_batch)
        graph_user_start_batch = np.array(graph_user_start_batch)
        graph_user_end_batch = np.array(graph_user_end_batch)
        user_vec_all_batch = np.array(user_vec_all_batch)
        user_vec_kind_batch = np.array(user_vec_kind_batch)
        loc_start_vev_batch = np.array(loc_start_vev_batch)
        loc_end_vec_batch = np.array(loc_end_vec_batch)
        hour_batch = np.array(hour_batch)
        week_batch = np.array(week_batch)
        yield graph_user_whole_batch,graph_user_start_batch,graph_user_end_batch,user_vec_all_batch,user_vec_kind_batch,loc_start_vev_batch,loc_end_vec_batch,hour_batch,week_batch





def generate_batch_data(train_data_path,node_vec_path,batch_size,shuffle):
    user_vec_files_path = glob.glob('../data/user_vec*')
    graph_user_files_path =  glob.glob('../data/vec_graph_user_*')
    train_data = load_train_data(train_data_path)
    node_vec_dict = load_node_vec_dict(node_vec_path)
    user_vec_dicts = get_user_vec_dicts(user_vec_files_path)
    graph_user_vec_dicts = get_graph_user_vec_dicts(graph_user_files_path)
    iterator = generate_sample(train_data,node_vec_dict,user_vec_dicts,graph_user_vec_dicts,shuffle = shuffle)
    return get_batch(iterator,batch_size)


# a = generate_batch_data(train_data_path,node_vec_path,batch_size=2)


node_vec_files = glob.glob('../data/loc_node_vec_order_combine*')
zip_path = zip(range(len(node_vec_files)),node_vec_files)
print 'we have those node vec files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
node_vec_path = node_vec_files[num]
train_data_path = '../data/train.csv'

print 'all data : 3214097'
skip_step = input('skip_step: ') # 300
n_round = input('n_round: ')
batch_size = input('batch_size: ')
learning_rate = input('learning_rate: ') #0.025
hour_embedding_size = 4
week_embedding_size = 4
loc_vec_size = int(node_vec_path.split('_dim_')[1])
graph_user_vec_size = int(glob.glob('../data/vec_graph_user_*')[0].split('_dim_')[1])
user_vec_size = loc_vec_size *2 
num_train_step = skip_step * n_round

# 3214097


def model(batch_gen,batch_size,learning_rate,hour_embedding_size,week_embedding_size):


    with tf.variable_scope("input"):
        graph_user_whole_vec = tf.placeholder(tf.float32, shape=[None,graph_user_vec_size],name = 'graph_user_whole_vec')
        graph_user_start_vec = tf.placeholder(tf.float32, shape=[None,graph_user_vec_size],name = 'graph_user_start_vec')
        graph_user_end_vec = tf.placeholder(tf.float32, shape=[None,graph_user_vec_size],name = 'graph_user_end_vec')
        user_vec_all = tf.placeholder(tf.float32, shape=[None,user_vec_size],name = 'user_vec_all')
        user_vec_kind = tf.placeholder(tf.float32, shape=[None,user_vec_size],name = 'user_vec_kind')
        loc_start_vec = tf.placeholder(tf.float32, shape=[None,loc_vec_size],name = 'loc_start_vec')
        loc_end_vec = tf.placeholder(tf.float32, shape=[None,loc_vec_size],name = 'loc_end_vec')
        hour = tf.placeholder(tf.int32, shape=[None],name = 'hour')
        week =tf.placeholder(tf.int32, shape=[None],name = 'week')
        keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')


    with tf.variable_scope("embedding"):
        hour_embedding = tf.Variable(tf.random_uniform([24, hour_embedding_size], -1.0, 1.0),name = 'hour_embedding')
        hour_index_embed = tf.nn.embedding_lookup(hour_embedding, hour)

        week_embedding = tf.Variable(tf.random_uniform([7, week_embedding_size], -1.0, 1.0),name = 'week_embedding')
        week_index_embed = tf.nn.embedding_lookup(week_embedding, week)     

    with tf.variable_scope("model_structure"):

         user_info = tf.concat([graph_user_whole_vec,graph_user_start_vec,graph_user_end_vec,user_vec_all,user_vec_kind],1,name = 'user_info')
         user_info_resize = fully_connected(user_info,num_outputs = 256,activation_fn = tf.nn.tanh)
         user_info_resize = fully_connected(user_info_resize,num_outputs = 128,activation_fn = tf.nn.tanh)
         user_info_resize = fully_connected(user_info_resize,num_outputs = 64,activation_fn = tf.nn.tanh)
         user_info_resize = fully_connected(user_info_resize,num_outputs = loc_vec_size,activation_fn = tf.nn.tanh)
         user_loc_start_mul = tf.multiply(user_info_resize,loc_start_vec)
         user_loc_start_mul = fully_connected(user_loc_start_mul,num_outputs = loc_vec_size*3,activation_fn = tf.nn.tanh)
         user_loc_start_mul = fully_connected(user_loc_start_mul,num_outputs = loc_vec_size*2,activation_fn = tf.nn.tanh)
         user_loc_start_mul = fully_connected(user_loc_start_mul,num_outputs = loc_vec_size,activation_fn = tf.nn.tanh)
         combine = tf.concat([loc_start_vec,graph_user_end_vec,graph_user_whole_vec,user_info_resize,user_loc_start_mul,hour_index_embed,week_index_embed],1,name ='combine')
         #combine = tf.nn.dropout(combine,keep_prob = keep_prob)
         result = fully_connected(combine , num_outputs = 128,activation_fn = tf.nn.tanh)
         result = fully_connected(combine , num_outputs = 256,activation_fn = tf.nn.tanh)
         result = fully_connected(result , num_outputs = 128,activation_fn = tf.nn.tanh)
         result = fully_connected(result , num_outputs = 64,activation_fn = tf.nn.tanh)
         result = fully_connected(result , num_outputs = 64,activation_fn = tf.nn.tanh)
         predict = fully_connected(result , num_outputs = loc_vec_size,activation_fn = None)


    #loss = tf.reduce_mean(tf.square(predict-loc_end_vec))
    loss = tf.reduce_mean(1-tf.diag_part(tf.matmul(predict,tf.transpose(loc_end_vec)))/(tf.norm(predict,axis=1)*tf.norm(loc_end_vec,axis=1)))


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #tf.train.RMSPropOptimizer
    # tf.train.AdamOptimizer

    loss_f_path = 'loss_record_learning_rate_' + str(learning_rate) + '_num_train_step_' + str(num_train_step) + '_batch_size_' + str(batch_size)
    loss_f = open(loss_f_path,'w')


    # Save Model Prepare Start
    export_dir = '../model/model_learning_rate_' + str(learning_rate) + '_num_train_step_' + str(num_train_step) + '_batch_size_' + str(batch_size)
    if os.path.exists(export_dir):
        os.rmdir(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    inputs = {'graph_user_whole_vec':tf.saved_model.utils.build_tensor_info(graph_user_whole_vec), 
              'graph_user_start_vec':tf.saved_model.utils.build_tensor_info(graph_user_start_vec), 
              'graph_user_end_vec':tf.saved_model.utils.build_tensor_info(graph_user_end_vec), 
              'user_vec_all': tf.saved_model.utils.build_tensor_info(user_vec_all), 
              'user_vec_kind': tf.saved_model.utils.build_tensor_info(user_vec_kind),
              'loc_start_vec': tf.saved_model.utils.build_tensor_info(loc_start_vec),
              'hour': tf.saved_model.utils.build_tensor_info(hour),
              'week': tf.saved_model.utils.build_tensor_info(week),
              'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob),
            }
    outputs = {'predict' : tf.saved_model.utils.build_tensor_info(predict)}

    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'sig_name')
    # Save Model Prepare End


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        loss_report_round = 0
        loss_list =[]
        t1 = time.time()
        for num in range(num_train_step):
            graph_user_whole_batch,graph_user_start_batch,graph_user_end_batch,user_vec_all_batch,user_vec_kind_batch,loc_start_vev_batch,loc_end_vec_batch,hour_batch,week_batch = next(batch_gen)
    
            loss_batch, _ = sess.run([loss, optimizer], 
                                        feed_dict={graph_user_whole_vec : graph_user_whole_batch,
                                                   graph_user_start_vec : graph_user_start_batch,
                                                   graph_user_end_vec : graph_user_end_batch,
                                                   user_vec_all : user_vec_all_batch, 
                                                   user_vec_kind : user_vec_kind_batch,
                                                   loc_start_vec : loc_start_vev_batch,
                                                   loc_end_vec : loc_end_vec_batch,
                                                   hour : hour_batch,
                                                   week : week_batch,
                                                   keep_prob : 0.5
                                                })
            if num == 1:
                print loss_batch
            total_loss += loss_batch

            if (num + 1) % skip_step == 0:
                loss_report_round += 1
                t2 = time.time()
                print 'Average loss at loss_report_round %d: %f , cost time : %d s'%(loss_report_round, total_loss * 1.0 / skip_step,t2-t1)
                t1 = time.time()
                loss_list.append(total_loss * 1.0 / skip_step)
                loss_f.write(str(loss_list[-1])+'\n')
                total_loss = 0.0

        loss_f.close()

        # Save Model
        builder.add_meta_graph_and_variables(sess,["model"],{'model_signature':signature})

    builder.save()


def main():
    batch_gen = generate_batch_data(train_data_path,node_vec_path,batch_size,shuffle = True)
    model(batch_gen,batch_size,learning_rate,hour_embedding_size,week_embedding_size)


if __name__ == '__main__':
    main()





