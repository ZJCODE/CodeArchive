import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
relation_train = pd.read_csv('./Data/relation_train.csv')
relation_test = pd.read_csv('./Data/relation_test.csv')
relation_train_label = pd.read_csv('./Data/relation_train_label.csv')



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
    
    
def prepare_vector_element_train(G,relation,label,filename):
    f = open('./Data/'+filename +'.txt','wb')
    for r,l in zip(relation.values,label.values):
        line = str(l[0]) + '|'
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


print 'test : '
prepare_vector_element_test(G,relation_test,'relation_test_vec_element')

print 'train original len '
train_original_len = len(relation_train)
print train_original_len

prepare_vector_element_train(G,relation_train,relation_train_label,'relation_train_vec_element_all_data')


'''
choose_len = int(train_original_len *0.9)

relation_train_choose = relation_train.ix[:choose_len,:]
relation_train_choose_val = relation_train.ix[choose_len:,:]

relation_train_label_choose = relation_train_label.ix[:choose_len,:]
relation_train_label_choose_val = relation_train_label.ix[choose_len:,:]


print 'train len'
print len(relation_train_label_choose)
print 'val len'
print len(relation_train_label_choose_val)

print 'train '
prepare_vector_element_train(G,relation_train_choose,relation_train_label_choose,'relation_train_vec_element')
print 'val'
prepare_vector_element_train(G,relation_train_choose_val,relation_train_label_choose_val,'relation_train_vec_element_val')
'''

'''
relation_train_choose = relation_train.ix[:2048,:]
relation_train_choose_val = relation_train.ix[:1024,:]

relation_train_label_choose = relation_train_label.ix[:2048,:]
relation_train_label_choose_val = relation_train_label.ix[:1024,:]

print 'train '
prepare_vector_element_train(G,relation_train_choose,relation_train_label_choose,'relation_train_vec_element_test')
print 'val'
prepare_vector_element_train(G,relation_train_choose_val,relation_train_label_choose_val,'relation_train_vec_element_val_test')
'''