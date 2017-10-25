# -*- coding: utf-8 -*-
import glob
import numpy as np
import os

user_files = glob.glob('../data/user_place*')
node_vec_files = glob.glob('../data/loc_node_vec_order_combine*')

zip_path = zip(range(len(node_vec_files)),node_vec_files)
print 'we have those node vec files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
node_vec_path = node_vec_files[num]

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

node_vec_dict = load_node_vec_dict(node_vec_path,skip_line = False,normalize=False)
node_vec_dim = int(node_vec_path.split('dim_')[1])

for f in user_files:
    print 'process %s'%(f)
    user_vec_file_path = f.split('places')[0]+'vec'+f.split('places')[1] 
    user_file = open(f,'r')
    user_vec = open(user_vec_file_path,'w')
    for line in user_file.readlines():
        user_start_vec = []
        user_end_vec = []
        start_end = [x.strip().split('-') for x in line.split('|')[1].split('\t')]
        for start,end in start_end:
            sv = node_vec_dict[start]
            ev = node_vec_dict[end]
            user_start_vec.append(sv)
            user_end_vec.append(ev)
        user_start_vec = np.array(user_start_vec).mean(0).tolist()
        user_end_vec = np.array(user_end_vec).mean(0).tolist()
        user_start_vec = [round(x,5) for x in user_start_vec]
        user_end_vec = [round(x,5) for x in user_end_vec]
        output_line = str(line.split('|')[0]) + ' ' + ' '.join(map(str,user_start_vec)) + ' ' + ' '.join(map(str,user_end_vec)) + '\n'
        user_vec.write(output_line)
    user_vec.close()


try:
    os.system('rm ../data/vec_graph*')
except:
    pass

graph_user_files = glob.glob('../data/graph_user*')

for f in graph_user_files:
    vec_path = f.split('graph')[0] + 'vec_graph' + f.split('graph')[1]
    graph_file = open(f,'r')
    graph_file.readline()
    user_vec = open(vec_path,'w')
    for line in graph_file.readlines():
        if line[0] != 'w':
            user_vec.write(line)
    user_vec.close()

try:
    os.system('rm ../data/user_place*')
except:
    pass

try:
    os.system('rm ../data/graph_user*')
except:
    pass



