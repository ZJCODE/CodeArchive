# -*- coding: utf-8 -*-
import networkx as nx
import glob
import numpy as np

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

def Cos_Similarity(x,M):
    x = np.array(x)
    M = np.array(M)
    if len(M.shape) == 2:
        cos_similarity = np.dot(M,x.T) / (np.linalg.norm(M,axis=1)*np.linalg.norm(x))
    else:
        cos_similarity = np.dot(M,x.T) / (np.linalg.norm(M)*np.linalg.norm(x))
    return cos_similarity



def sim_top_3(v,M,M_name):
    v = np.array(v)
    M = np.array(M)
    M_name = np.array(M_name)
    dist = 1-Cos_Similarity(v,M)
    sim_order = M_name[np.argsort(dist)].tolist()
    if len(sim_order)>=3:
        return sim_order[:3]
    else:
        sim_order.append(sim_order[-1])
        sim_order.append(sim_order[-1])
        return sim_order[:3]

node_vec_files = glob.glob('../data/loc_node_vec_order_combine*')
zip_path = zip(range(len(node_vec_files)),node_vec_files)
print 'we have those node vec files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
node_vec_path = node_vec_files[num]
node_vec_dict = load_node_vec_dict(node_vec_path)
loc_vec_size = int(node_vec_path.split('_dim_')[1])


G = nx.Graph()
f = open('../data/loc_start_end_pair','r')
for line in f.readlines():
    edge = [line.strip().split('\t')[:2]]
    G.add_edges_from(edge)
f.close()


# load predict loc vec files 
predcit_loc_end_all = glob.glob('../data/predcit_loc_end_*')
zip_path = zip(range(len(predcit_loc_end_all)),predcit_loc_end_all)
print 'we have those predcit files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
predcit_loc_end = predcit_loc_end_all[num]


predict = open(predcit_loc_end,'r')
test_data = open('../data/test.csv','r')
_ = test_data.readline()
submission_path = '../data/submission' + predcit_loc_end.split('_loc_end')[1] + '.csv'
submission = open(submission_path,'w')

all_loc = G.nodes()
i = 0
for test_line,line in zip(test_data.readlines(),predict.readlines()):
    i += 1
    if i % 10000 == 0:
        print 'process %d'%(i)
    pred_vec = map(float,line.split('|to|')[1].strip().split(' '))
    loc_start = line.split('|to|')[0].split('-')[-1]
    order_id = test_line.strip().split(',')[0]
    try:
        neighbors = nx.neighbors(G,loc_start)
    except:
        neighbors = [loc for loc in all_loc if loc.startswith(loc_start[:5])]
        if len(neighbors) < 3:
            neighbors = [loc for loc in all_loc if loc.startswith(loc_start[:4])]

    if len(neighbors) < 3 :
        neighbors = neighbors + [loc for loc in all_loc if loc.startswith(loc_start[:5])]
        if len(neighbors) < 3 :
            neighbors = neighbors + [loc for loc in all_loc if loc.startswith(loc_start[:4])]

    if len(neighbors) < 3:
        neighbors = [loc_start] *3

    neighbors_vec = []
    for n in neighbors:
        try:
            neighbors_vec.append(node_vec_dict[n])
        except:
            neighbors_vec.append([100]*loc_vec_size)
    sim_result = sim_top_3(v = pred_vec,M = neighbors_vec,M_name = neighbors)

    submission.write(order_id+','+','.join(sim_result)+'\n')

submission.close()
predict.close()
test_data.close()







