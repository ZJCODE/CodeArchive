# Input Two Columns of Data
# First Columns User
# Sencond Columns Item

import pandas as pd
from collections import Counter
import numpy as np

Data = pd.read_csv('./Data/data.csv',names = ['user','item'])

userid = Data.user
movieid = Data.item

rating = abs(np.random.randn(len(Data))*10).astype('int')+1
timestamp = abs(np.random.randn(len(Data)))

Pair_Data = pd.DataFrame({'userid':userid,'movieid':movieid,'rating':rating,'timestamp':timestamp})
Pair_Data = Pair_Data[['userid', 'movieid', 'rating', 'timestamp']]



# select user 

select_num = 0
user_count = Counter(Pair_Data.userid.values)    
user_select = [x[0] for x in user_count.items() if x[1] >= select_num ]       
user_select_set = set(user_select)           
tag = [1 if user in user_select_set else 0 for user in Pair_Data.userid.values]    
Pair_Data['tag'] = tag          
Pair_Data_Select = Pair_Data[Pair_Data.tag == 1]  

Pair_Data_Select = Pair_Data_Select.ix[:,:4]

# output data 

print 'user'
print len(set(Pair_Data_Select.userid))
print 'item'
print len(set(Pair_Data_Select.movieid))
print 'pairs'
print len(Pair_Data_Select)
print 'pair orig'
print len(Pair_Data)


Pair_Data_Select.to_csv('./Data/data_for_split.csv',index =False)