# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:16:13 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from Net import GenerateNetwork,GenerateNetworkWithWeight,DrawGraph,GetCoreSubNetwork,CommunityDetection
from collections import Counter
import  matplotlib.pyplot as plt

def Import_Obj(File):    
    import pickle
    File_Name = File+'.pkl'
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)
    

def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()    

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L

def GenerateDate(year,month,day):
    return pd.datetime(year,month,day).date()



def ExportEdgesToGephi(relation):    
    R = pd.DataFrame(relation,columns=['Source','Target'])
    R.to_csv('edges.csv',index=False)
    
def ExportNodeCategoryToGephi(nodes_category):    
    nodes_category.to_csv('node_category.csv',index=False)

def MapLocation(location,place):
    if place == 'Queensland':
        hhs_loc = ['Cairns', 'Townsville', 'Mackay', 'Fitzroy', 'Wide Bay',
           'Sunshine Coast', 'Brisbane', 'Darling Downs', 'Moreton',
           'Gold Coast']
        
        if location.startswith('Brisbane'):
            location = 'Brisbane'
        if location.startswith('Moreton'):
            location = 'Moreton'
        
        if location not in hhs_loc:
            return None
        else:        
            return location
    else:
        return location

def GetMoveInWhere(Move,place):
    '''
    specific location's place
    '''
    if place == None:
        MoveInWhere = Move
    else:        
        MoveInWhere = [a for a in Move if (place in a[1] and place in a[2])]
    users = [a[0] for a in MoveInWhere]
    pairs = [(MapLocation(a[1].split(',')[0],place),MapLocation(a[2].split(',')[0],place)) for a in MoveInWhere]
    DfMove = pd.DataFrame({'user':users,'pairs':pairs})
    DfMove['same'] = [0 if l[0]==l[1] else 1 for l in DfMove.pairs]
    DfMove = DfMove[DfMove.same == 1]
    return DfMove



def FliterFlu(week_move,week_user_flu_state,place=None):
    
    weeks = sorted(week_move.keys())
    flu_pair = []
    actual_day_list = []
    for w in weeks:
        move = week_move[w]
        MoveSomeWhere = GetMoveInWhere(move,place)
        flu_state = week_user_flu_state[w]
        flu_users = flu_state[0]
        actual_day_list.append(flu_state[1])
        Judge = [1 if user in flu_users else 0 for user in MoveSomeWhere.user]
        #Judge = [1 for user in MoveSomeWhere.user]
        MoveSomeWhere['Judge'] = Judge
        flu_related_data = MoveSomeWhere[MoveSomeWhere.Judge==1]
        flu_pair.append(flu_related_data.pairs.values)
        week_flu_pair = dict(zip(weeks,flu_pair))
    return week_flu_pair,actual_day_list

def CountInWhere(week_flu_pair):
    count_destination_list=[]
    weeks = sorted(week_flu_pair.keys())
    for w in weeks:
        pair = week_flu_pair[w]
        destination = [d[1] for d in pair]
        count_destination = Counter(destination)
        count_destination_list.append(count_destination)
    week_count_destination = dict(zip(weeks,count_destination_list))    
    return week_count_destination
    
def Normalize(x):
    x = np.array(x)
    return (x-x.mean())*1.0 / x.std()    
    
week_move = Import_Obj('./Data/week_move')
week_user_flu_state = Import_Obj('./Data/week_user_flu_state')
Queensland_Flu = pd.read_csv('./Data/Queensland2015.csv')
week_flu_pair,actual_day_list = FliterFlu(week_move,week_user_flu_state,'Queensland')

def AnalysisCompare(Queensland_Flu,week_flu_pair,actual_day_list,place,n,r):
    '''
    place = ['Cairns', 'Townsville', 'Mackay', 'Fitzroy', 'Wide Bay',
   'Sunshine Coast', 'Brisbane', 'Darling Downs', 'Moreton',
   'Gold Coast']
    '''
    m = CountInWhere(week_flu_pair)
    weeks = sorted(week_flu_pair.keys())
    
    l=[]
    for w in weeks:
        l.append(m[w][place])
        
    l = np.array(l)*1.0/np.array(actual_day_list)
            
        
    p = Queensland_Flu[place].values    
    
        
    plt.plot(np.r_[np.zeros(n),l[:19]]*r,'.-')
    plt.plot(p[:19],'*-')
    #plt.xlim([1.5,20.5])
    plt.legend(['l','p'])
    
    
    
def TestWeekMovePattern(y=2015,m=1,d=13):    
    week_move = Import_Obj('./Data/week_move')
    relation = []
    G = GenerateNetwork(relation,direct = True)
    SubG = GetCoreSubNetwork(G,0,100,'No')
    n,e = CommunityDetection(SubG,3,with_label=True,with_arrow=True)
    ExportEdgesToGephi(e)
    ExportNodeCategoryToGephi(n)

'''
WG = GenerateNetworkWithWeight(edge_with_weight,direct=True)    
WG.in_degree(WG.nodes()[20],weight='weight')
'''


    
    
    
