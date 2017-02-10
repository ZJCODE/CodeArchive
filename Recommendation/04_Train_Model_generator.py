'''
import pandas as pd

movies = pd.read_table('./Data/movies.dat',sep='::',engine='python',names=['movieid', 'title', 'genre']).set_index('movieid')
users = pd.read_table('./Data/users.dat',sep='::',engine='python',names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')

n_movies = movies.shape[0]
n_users = users.shape[0]
'''

import pandas as pd
import numpy as np
from sklearn import dummy, metrics, cross_validation, ensemble
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras import initializations
 





'''
f = open('./Data/relation_train_vec_element.txt')

lines = f.readlines()
'''

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
    label,user_item,user2item_neighbors,item2user_neighbors = line.strip().split('|')
    label = int(label)
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
    return user , item , user2item_neighbors_vec , item2user_neighbors_vec , label

'''
max_user = 0
max_item = 0
max_u_v = 0
max_i_v = 0


user_vecs = []
item_vecs =[]
userids = []
itemids = []
labels = []


for line in lines:
    line = line.strip()
    user,item,user2item_neighbors_vec,item2user_neighbors_vec,label = deal_with_line(line)
    if user > max_user:
        max_user = user
    if item > max_item:
        max_item = item
    if max(user2item_neighbors_vec) > max_u_v:
        max_u_v = max(user2item_neighbors_vec)
    if max(item2user_neighbors_vec) > max_i_v:
        max_i_v = max(item2user_neighbors_vec)
    user_vecs.append(user2item_neighbors_vec)
    item_vecs.append(item2user_neighbors_vec)
    userids.append(user)
    itemids.append(item)
    labels.append(label)
    
print max_user
print max_item
print max_u_v
print max_i_v	
    
user_vecs = np.array(user_vecs)
item_vecs = np.array(item_vecs)
userids = np.array(userids)
itemids = np.array(itemids)
labels = np.array(labels)

print 'user_vecs shape '
print user_vecs.shape
print 'item_vecs shape '
print item_vecs.shape
'''

'''
y = np.zeros((len(labels),2))
y[range(len(labels)),labels] = 1
'''

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(n_users,n_items,embedding_dim,nb_layer):
    
    userid_input = keras.layers.Input(shape=([1]),dtype='int32')
    userid_vec = keras.layers.Flatten()(keras.layers.Embedding(input_dim=n_users, output_dim =embedding_dim,init = init_normal)(userid_input))

    movieid_input = keras.layers.Input(shape=([1]),dtype='int32')
    movieid_vec = keras.layers.Flatten()(keras.layers.Embedding(input_dim=n_items,output_dim =embedding_dim,init = init_normal)(movieid_input))

    element_multiple = keras.layers.merge([userid_vec, movieid_vec], mode='mul')
    element_multiple = keras.layers.Dropout(0.5)(element_multiple)

    movie_input = keras.layers.Input(shape=(MAX_ITEM_NEIGHBORS_LEN,),dtype='int32')
    #movie_input = keras.layers.Input(shape=([1]),dtype='int32')
    #movie_input = keras.layers.Input(shape=(n_items,),dtype='int32')
    movie_vec = keras.layers.Embedding(input_dim=n_items , output_dim=embedding_dim,init = init_normal)(movie_input)
    movie_vec = Conv1D(128, 5, activation='tanh')(movie_vec)
    movie_vec = MaxPooling1D(5)(movie_vec)
    movie_vec = keras.layers.Flatten()(movie_vec)
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)

    user_input = keras.layers.Input(shape=(MAX_USER_NEIGHBORS_LEN,),dtype = 'int32')
    #user_input = keras.layers.Input(shape=([1]),dtype = 'int32')
    #user_input = keras.layers.Input(shape=(n_users,),dtype = 'int32')
    user_vec = keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim,init = init_normal)(user_input)
    user_vec = Conv1D(128, 5, activation='tanh')(user_vec)
    user_vec = MaxPooling1D(5)(user_vec)
    user_vec = keras.layers.Flatten()(user_vec)
    user_vec = keras.layers.Dropout(0.5)(user_vec)

    input_vecs = keras.layers.merge([movie_vec, element_multiple ,user_vec], mode='concat')
    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
    nn = keras.layers.normalization.BatchNormalization()(nn)

    for i in range(1,nb_layer):
        nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
        nn = keras.layers.normalization.BatchNormalization()(nn)

    nn = keras.layers.Dense(128, activation='relu')(nn)

    #result = keras.layers.Dense(2, activation='softmax')(nn)

    result = keras.layers.Dense(1, activation='sigmoid')(nn)



    model = kmodels.Model([userid_input, movieid_input,user_input,movie_input], result)
    model.compile('adam', 'binary_crossentropy',metrics=['accuracy'])

    return model






def generate_arrays_from_file(path):

    while 1:
        f = open(path)
        user_vecs = []
        item_vecs =[]
        userids = []
        itemids = []
        labels = []
        i = 0
        for line in f:
            i = i+1
            try:            
                user , item , user2item_neighbors_vec , item2user_neighbors_vec , label = deal_with_line(line)
                user_vecs.append(user2item_neighbors_vec)
                item_vecs.append(item2user_neighbors_vec)
                userids.append(user)
                itemids.append(item)
                labels.append(label)
            except:
                f.close()
                break
                pass



            if i % 1024 == 0 :
                user_vecs = np.array(user_vecs)
                item_vecs = np.array(item_vecs)
                userids = np.array(userids)
                itemids = np.array(itemids)
                labels = np.array(labels)
                yield ([userids , itemids , user_vecs , item_vecs], labels)
                user_vecs = []
                item_vecs =[]
                userids = []
                itemids = []
                labels = []



def generate_arrays_from_file_val(path):
    while 1:
        f = open(path)
        user_vecs = []
        item_vecs =[]
        userids = []
        itemids = []
        labels = []
        i = 0
        for line in f:
            i = i+1
            try:
                user , item , user2item_neighbors_vec , item2user_neighbors_vec , label = deal_with_line(line)
                user_vecs.append(user2item_neighbors_vec)
                item_vecs.append(item2user_neighbors_vec)
                userids.append(user)
                itemids.append(item)
                labels.append(label)
            except:
                f.close()
                break
                pass


            if i % 1024 == 0 :
                user_vecs = np.array(user_vecs)
                item_vecs = np.array(item_vecs)
                userids = np.array(userids)
                itemids = np.array(itemids)
                labels = np.array(labels)
                yield ([userids , itemids , user_vecs , item_vecs], labels)
                user_vecs = []
                item_vecs =[]
                userids = []
                itemids = []
                labels = []





def evaluate(model):
    
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
        
        pred_prob = model.predict([userids,itemids,user_vecs,item_vecs])
        Z = zip(user_item_pair,pred_prob)
        Z.sort(key = lambda x : x[1] , reverse = True)
        pair_list,_ = zip(*Z)
        test_index = pair_list.index(test_user_item_pair) + 1
        
        test_index_list.append(test_index)
        

    test_pred_loc = pd.DataFrame({'location':test_index_list})

    model_hr = []

    for n in range(1,11):
        model_pred_n  =  [1 if x <=n else 0 for x in test_pred_loc.location.values]
        model_hr.append(sum(model_pred_n) * 1.0  / len(test_pred_loc))

    return model_hr[-1]


n_items = 3706 
n_users = 6040
embedding_dim = 32
nb_layer = 3

model = get_model(n_users,n_items,embedding_dim,nb_layer)
print 'model prepared !'

hr = 0

hr_list = []

for epoch in range(20):
    print 'Epoch %d'%(epoch+1)
    model.fit_generator(generate_arrays_from_file('./Data/relation_train_vec_element_all_data.txt'),
                        samples_per_epoch=4970496, 
                        nb_epoch=1)

    hr_at_10 = evaluate(model)
    hr_list.append(hr_at_10)
    print hr_at_10
    if hr_at_10 > hr :
        json_string = model.to_json()
        open('my_model_architecture_all.json','w').write(json_string)
        model.save_weights('my_model_weights_all.h5')
        print 'save model'
        hr = hr_at_10

def Save_List(List,Name):
    File = Name + '.txt'
    pd.DataFrame({Name:List}).to_csv(File,encoding='utf8',header=True,index = False)
   

Save_List(hr_at_10,'hr_at_10')




'''
#a_userid, b_userid, a_itemid , b_itemid , a_user_vec , b_user_vec , a_item_vec ,b_item_vec ,a_y, b_y = cross_validation.train_test_split(userids, itemids, user_vecs,item_vecs, y,test_size=0.1)
a_userid, b_userid, a_itemid , b_itemid , a_user_vec , b_user_vec , a_item_vec ,b_item_vec ,a_y, b_y = cross_validation.train_test_split(userids, itemids, user_vecs,item_vecs, labels,test_size=0.1)

model.fit([a_userid,a_itemid,a_user_vec,a_item_vec], a_y, 
                         nb_epoch=20, 
                         validation_data=([b_userid,b_itemid,b_user_vec,b_item_vec], b_y),batch_size = 1024)


model.fit([userids,itemids,user_vecs,item_vecs],y,nb_epoch = 5)
'''






