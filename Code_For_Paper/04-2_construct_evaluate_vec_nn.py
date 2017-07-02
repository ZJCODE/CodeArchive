import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import networkx as nx
import community

relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
relation_test = pd.read_csv('./Data/relation_test.csv')
all_item = pd.read_csv('./Data/all_item.csv')

all_item = all_item.item.values

print 'item len'
print len(all_item)

N = 99
Max_Num_Neighbor =50

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
    if len(un_interact_item) < N:
        un_interact_item = un_interact_item * (int(N / len(un_interact_item))+1)
    un_interact_item_choose = un_interact_item[:N]
    uninteract_user_list += [user] * N
    uninteract_item_list += un_interact_item_choose
    
uninteract = pd.DataFrame(np.array([uninteract_user_list,uninteract_item_list]).T , columns = ['user','item'])



def Cos(v1,M):
    return np.dot(v1,M.T)/(np.linalg.norm(v1)*np.linalg.norm(M.T,axis=0))

user_index = [int(x.split('_')[1]) for x in relation_train_positive.user]
item_index = [int(x.split('_')[1]) for x in relation_train_positive.item]
M = np.zeros([max(user_index)+1,max(item_index)+1])
for u,i in zip(user_index , item_index):
    M[u,i] = 1

def get_neighbor(node,M,k=50):
    kind,i = node.split('_')
    i = int(i)
    if kind == 'user':        
        cos_score = Cos(M[i,:],M)
        cos_index_zip = zip(range(len(cos_score)),cos_score)
        cos_index_zip.sort(key= lambda x : x[1],reverse=True)
        neighbor = [x[0] for x in cos_index_zip[:k]]
        neighbor = [kind+'_'+str(x) for x in neighbor]
    if kind == 'item':
        M = M.T
        cos_score = Cos(M[i,:],M)
        cos_index_zip = zip(range(len(cos_score)),cos_score)
        cos_index_zip.sort(key= lambda x : x[1],reverse=True)
        neighbor = [x[0] for x in cos_index_zip[:k]]
        neighbor = [kind+'_'+str(x) for x in neighbor]
    return neighbor

user_node = []
user_neighbor = []
item_node = []
item_neighbor =[]

for u in list(set(relation_train_positive.user)):
    user_node.append(u)
    user_neighbor.append(get_neighbor(u,M))
user_neighbor_dict = dict(zip(user_node,user_neighbor))

for i in list(set(relation_train_positive.item)):
    item_node.append(i)
    item_neighbor.append(get_neighbor(i,M))
item_neighbor_dict = dict(zip(item_node,item_neighbor))

print 'neighbor dict prepared !'



def prepare_vector_element_test(relation,filename,M):
    f = open('./Data/'+filename +'.txt','wb')
    for r in relation.values:
        line = ''
        user,item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'


    
        try:
            item2user_neighbor = user_neighbor_dict[user]
            np.random.shuffle(item2user_neighbor)
            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]

        except:
            item2user_neighbor = []
        
        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor + '|'


        try:
            user2item_neighbor = item_neighbor_dict[item]
            np.random.shuffle(user2item_neighbor)
            if len(user2item_neighbor) > Max_Num_Neighbor:
                user2item_neighbor = user2item_neighbor[:Max_Num_Neighbor]
        except:
             user2item_neighbor = []
             
        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor 

        f.writelines(line)
        f.writelines('\n')
    f.close()
    
print 'test positive : '
print 'test len'
print len(relation_test)
prepare_vector_element_test(relation_test,'relation_test_vec_element',M)
print 'test uninteract : '
print 'uninteract len'
print len(uninteract)
prepare_vector_element_test(G,uninteract,'uninteract_vec_element')
