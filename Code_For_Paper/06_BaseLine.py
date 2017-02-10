import numpy as np
import pandas as pd
from collections import Counter
from math import log

relation_train = pd.read_csv('./Data/relation_train.csv')
relation_train_label = pd.read_csv('./Data/relation_train_label.csv')

train_choose_data_label = zip(relation_train.values,relation_train_label.values)
train_choose_data_interact = [x[0] for x in train_choose_data_label if x[1] == 1]

item_count = Counter([int(x[1].split('_')[1]) for x in train_choose_data_interact])

MAX_USER_NEIGHBORS_LEN = 50
MAX_ITEM_NEIGHBORS_LEN = 50

f = open('./Data/relation_test_vec_element.txt','r')
test_vec_lines = f.readlines()
f.close()

f = open('./Data/uninteract_vec_element.txt','r')
uninteract_vec_lines = f.readlines()
f.close()



def pad_sequences(X,maxlen):
    if len(X) > maxlen:
        v = np.array(X[:maxlen])
    else:
        v = np.zeros(maxlen)
        for i,x in enumerate(X):
            v[i] = x
    v = v.astype('int')
    return v

def deal_with_line(line):
    line = line.strip()
    user_item,user2item_neighbors,item2user_neighbors = line.strip().split('|')
    user,item = [int(x) for x in user_item.split(',')]
    try:
        item2user_neighbors = [int(x) for x in item2user_neighbors.split(',')]
    except:
        item2user_neighbors = []
    try:
        user2item_neighbors = [int(x) for x in user2item_neighbors.split(',')]
    except:
        user2item_neighbors = []
    user2item_neighbors_vec = pad_sequences(user2item_neighbors,maxlen=MAX_USER_NEIGHBORS_LEN)
    item2user_neighbors_vec = pad_sequences(item2user_neighbors,maxlen=MAX_ITEM_NEIGHBORS_LEN)
    return user , item , user2item_neighbors_vec , item2user_neighbors_vec


test_len = len(test_vec_lines)


test_index_list = []
baseline_index_list = []
for i in range(test_len):
    user_vecs = []
    item_vecs =[]
    userids = []
    itemids = []
    user_item_pair = []
    test = test_vec_lines[i].strip()
    user,item,user2item_neighbors_vec,item2user_neighbors_vec = deal_with_line(test)
    test_user_item_pair = str(user)+'_'+str(item)
    userids.append(user)
    itemids.append(item)
    user_item_pair.append(str(user)+'_'+str(item))
    user_vecs.append(user2item_neighbors_vec)
    item_vecs.append(item2user_neighbors_vec)
    uninteract = uninteract_vec_lines[i*99:(i+1)*99]
    for l in uninteract:
        l = l.strip()
        user,item,user2item_neighbors_vec,item2user_neighbors_vec = deal_with_line(l)
        userids.append(user)
        itemids.append(item)
        user_item_pair.append(str(user)+'_'+str(item))
        user_vecs.append(user2item_neighbors_vec)
        item_vecs.append(item2user_neighbors_vec)
    
    user_vecs = np.array(user_vecs)
    item_vecs = np.array(item_vecs)
    userids = np.array(userids)
    itemids = np.array(itemids)
    

    baseline_count = []
    for itemid in itemids:
        try:
            baseline_count.append(item_count[itemid])
        except:
            baseline_count.append(0)
    Zb = zip(user_item_pair,baseline_count)
    Zb.sort(key = lambda x : x[1] , reverse = True)
    pair_list2,_ = zip(*Zb)
    baseline_index = pair_list2.index(test_user_item_pair) + 1
    baseline_index_list.append(baseline_index)

baseline_pred = pd.DataFrame({'location':baseline_index_list})


baseline_hr = []

for n in range(1,11):
    baseline_pred_n = [1 if x <=n else 0 for x in baseline_pred.location.values]
    print sum(baseline_pred_n)
    baseline_hr.append(sum(baseline_pred_n) * 1.0 / len(baseline_pred))

model_NDCG_1_10 = []

for n in range(1,11):

    model_NDCG_at10 = []
    for p in baseline_pred.location.values:
        if p <= n :
            model_NDCG_at10.append(1.0/(log(1+p) / log(2)))
        else:
            model_NDCG_at10.append(0)

    model_NDCG = np.mean(model_NDCG_at10)
    model_NDCG_1_10.append(model_NDCG)


print 'ndcg_1-10'
print model_NDCG_1_10
print 'hr_1-10'
print baseline_hr