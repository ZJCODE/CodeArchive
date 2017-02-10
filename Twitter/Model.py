# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:46:26 2016

@author: ZJun
"""

from twitter_function import Import_Obj,GenerateDate,hinton



df_flu_related_twitter_in_place_loc = Import_Obj('./DF_Result/df_flu_related_twitter_in_place_loc')
df_twitter_in_place_loc = Import_Obj('./DF_Result/df_twitter_in_place_loc')
df_move_destination_in_place_loc = Import_Obj('./DF_Result/df_move_destination_in_place_loc')
df_flu_related_move_destination_in_place_loc = Import_Obj('./DF_Result/df_flu_related_move_destination_in_place_loc')
df_real_flu = Import_Obj('./DF_Result/df_real_flu')
df_where_to_where = Import_Obj('./DF_Result/df_where_to_where')
df_flu_where_to_where = Import_Obj('./DF_Result/df_flu_where_to_where')



def GetTransferProbability(df_where_to_where,df_twitter_in_place_loc):
    
    all_where_to_where = df_where_to_where.values[0]*0 
    
    for w2w in df_where_to_where.values:
        all_where_to_where += w2w
    
    # all_where_to_where  i to j  [i,j]
    
    num_all = df_twitter_in_place_loc.sum(0)
    num_move = all_where_to_where.sum(1)
    num_stay = num_all - num_move
    
    all_where_to_where_with_stay = all_where_to_where.copy()
    for loc in all_where_to_where_with_stay.columns:
        all_where_to_where_with_stay.loc[loc,loc] = num_stay[loc]
    
    
    all_where_to_where_T = all_where_to_where.T  # [i,j]  j to i
    
    P = (all_where_to_where_T / all_where_to_where_T.sum()).T # [i,j]  i to j
    
    
    
    
    all_where_to_where_with_stay_T = all_where_to_where_with_stay.T  # [i,j]  j to i
    
    P_with_stay = (all_where_to_where_with_stay_T / all_where_to_where_with_stay_T.sum()).T # [i,j]  i to j

    return P,P_with_stay
    
    
    
'''    
    
x0 = df_real_flu.ix[0,:].as_matrix()    
p = np.matrix(P_with_stay.values)

A = p
for _ in range(100):
    A = A*p

A = pd.DataFrame(A,columns=df_flu_where_to_where.values[0].index,index = df_flu_where_to_where.values[0].index)

hinton(A)
'''