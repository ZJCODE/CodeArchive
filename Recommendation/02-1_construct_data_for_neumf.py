import pandas as pd
import networkx as nx
import numpy as np

relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')

relation_test = pd.read_csv('./Data/relation_test.csv')


train_user = [x.split('_')[1] for x in relation_train_positive.user.values]
train_item = [x.split('_')[1] for x in relation_train_positive.item.values]
train_rating = [1]*len(relation_train_positive)

train = pd.DataFrame({'user':train_user,'item':train_item,'rating':train_rating})[['user','item','rating']]

train.to_csv('./Data/ml-1m.train.rating',sep = '\t' , header = False , index = False)



test_user = [x.split('_')[1] for x in relation_test.user.values]
test_item = [x.split('_')[1] for x in relation_test.item.values]
test_rating = [1]*len(relation_test)

test_df = pd.DataFrame({'user':test_user,'item':test_item,'rating':test_rating})[['user','item','rating']]

test_df.to_csv('./Data/ml-1m.test.rating',sep = '\t',header = False , index = False)



N = 99


G_all = nx.Graph()
G_all.add_edges_from(relation_train_positive.values)
G_all.add_edges_from(relation_test.values)

test = relation_test.values

all_item = set (relation_train_positive.item.values) & set(relation_test.item.values)



f = open('./Data/ml-1m.test.negative','wb')
i = 0
for user,item in test:
    i = i + 1
    print 'process' + str(i)
    interact_item = nx.neighbors(G_all,user)
    un_interact_item = list(set(all_item) - set(interact_item))
    np.random.shuffle(un_interact_item)
    un_interact_item_choose = un_interact_item[:N]

    line = '(' + str(user.split('_')[1]) + ','+ str(item.split('_')[1]) + ')'

    for uninteract in un_interact_item_choose:
    	line = line + '\t' + str(uninteract.split('_')[1])
    f.writelines(line)
    f.writelines('\n')

f.close()