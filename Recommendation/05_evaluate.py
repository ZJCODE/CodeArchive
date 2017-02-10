

import numpy as np
import pandas as pd

def evaluate(model)

    f = open('./Data/relation_test_vec_element.txt','r')
    test_vec_lines = f.readlines()
    f.close()

    f = open('./Data/uninteract_vec_element.txt','r')
    uninteract_vec_lines = f.readlines()
    f.close()


    MAX_USER_NEIGHBORS_LEN = 200
    MAX_ITEM_NEIGHBORS_LEN = 300

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
        user2item_neighbors_vec = pad_sequences(user2item_neighbors,maxlen=MAX_USER_NEIGHBORS_LEN)#user#to_several_hot(user,user2item_neighbors,n_users)
        item2user_neighbors_vec = pad_sequences(item2user_neighbors,maxlen=MAX_ITEM_NEIGHBORS_LEN)#item#to_several_hot(item,item2user_neighbors,n_items)
        return user , item , user2item_neighbors_vec , item2user_neighbors_vec


    test_len = len(test_vec_lines)


    test_index_list = []
    baseline_index_list = []
    for i in range(test_len):
        print 'precess ' + str(i)
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
        uninteract = uninteract_vec_lines[i*100:(i+1)*100]
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
        
        pred_prob = model.predict([userids,itemids,user_vecs,item_vecs])#[:,1]
        Z = zip(user_item_pair,pred_prob)
        Z.sort(key = lambda x : x[1] , reverse = True)
        pair_list,_ = zip(*Z)
        test_index = pair_list.index(test_user_item_pair) + 1
        
        test_index_list.append(test_index)
        


        
    test_pred_loc = pd.DataFrame({'location':test_index_list})


    model_hr = []

    for n in range(1,11):
        model_pred_n  =  [1 if x <=n else 0 for x in test_pred_loc.location.values]
        print sum(model_pred_n)
        model_hr.append(sum(model_pred_n) * 1.0  / len(test_pred_loc))

    return model_hr[-1]