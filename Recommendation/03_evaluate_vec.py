import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
relation_test = pd.read_csv('./Data/relation_test.csv')
all_item = pd.read_csv('./Data/all_item.csv')

all_item = all_item.item.values

print 'item len'
print len(all_item)

N = 100


G_all = nx.Graph()
G_all.add_edges_from(relation_train_positive.values)
G_all.add_edges_from(relation_test.values)

test_user = relation_test.user.values

uninteract_user_list = []
uninteract_item_list = []

for user in test_user:
    interact_item = nx.neighbors(G_all,user)
    un_interact_item = list(set(all_item) - set(interact_item))
    np.random.shuffle(un_interact_item)
    un_interact_item_choose = un_interact_item[:N]
    uninteract_user_list += [user] * N
    uninteract_item_list += un_interact_item_choose
    
uninteract = pd.DataFrame(np.array([uninteract_user_list,uninteract_item_list]).T , columns = ['user','item'])

G = nx.Graph()
G.add_edges_from(relation_train_positive.values)

def prepare_vector_element_test(G,relation,filename):
    f = open('./Data/'+filename +'.txt','wb')
    for r in relation.values:
        line = ''
        user,item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'
        try:
            user2item_neighbor = nx.neighbors(G,item)
        except:
             user2item_neighbor = []
             
        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor + '|'
    
        try:
            item2user_neighbor = nx.neighbors(G,user)
        except:
            item2user_neighbor = []
        
        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor
        
        f.writelines(line)
        f.writelines('\n')
    f.close()
    
print 'test positive : '
print 'test len'
print len(relation_test)
prepare_vector_element_test(G,relation_test,'relation_test_vec_element')
print 'test uninteract : '
print 'uninteract len'
print len(uninteract)
prepare_vector_element_test(G,uninteract,'uninteract_vec_element')
