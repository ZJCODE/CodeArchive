# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from PowLaw import *
import networkx as nx
from Tools import *

#rl=[]
#for num_person in range(20,1000):
num_interest=20
num_person = input('Number of people in this Network( > 20) : ')
initial_num_interest_person_max = 10
same_interest_threshold = 3



Interest_Space=range(num_interest)# each number represents a interest

# [ID,[],personal_interest_space,influence] stand for [ID,[Connetction],[Personal Interest Space],[influence]]

Peoples=[]

for i in range(num_person):
    num_interest_each_person_initial=random.choice(range(1,initial_num_interest_person_max))
    #random.shuffle(Interest_Space)
    #personal_interest_space = Interest_Space[:num_interest_each_person_initial]
    personal_interest_space=random.sample(Interest_Space,num_interest_each_person_initial)
    Peoples.append([i,[],personal_interest_space])
    
ID=[]# contral preferential attachment
# if someone build links his id will be add to ID list again
# for example initial ID=[1,2,3,4,5] ,id=3'person reply to one post then ID list becomes [1,2,3,4,5,3]
# so his probability of being chosed next time will increase to 2/6 comparing to previous probability which is 1/5


def choose_influence_interest(Influence):
    table = Counter(Influence).items()
    influence_interest = [i[0] for i in table if i[1] >= influence_threshold]
    return influence_interest
    
    
def Add_Relation(Peoples,id_1,id_2):
    Peoples[id_1][1].append(id_2)
    Peoples[id_2][1].append(id_1)
    return None  
  
def Intersection_List(Set_List):
    S = Set_List[0]
    for i in Set_List[1:]:
        S = S & i
    return S
  
def Get_ID(ID_List):
    ID_List = set(ID_List)
    ID = []
    for i in range(len(ID_List)):
        ID += list(ID_List.pop())
    return ID
        

# Initial
initial_num = 20
ID_List = []

for i in range(initial_num):
    [p1,p2] = random.sample(range(initial_num),2)
    Add_Relation(Peoples,p1,p2)
    ID_List.append((p1,p2))
ID = Get_ID(ID_List)
    
# Growth

m = 13
influence_threshold = 5

for new_id in range(initial_num,num_person):
    new_id_interest = Peoples[new_id][2]

    Influencer_List = []
    # preferential attachment
    Potential_IDs = random.sample(ID,m)
    # Interest Space contral
    for p_id in Potential_IDs:
        
        p_id_interest = Peoples[p_id][2]
        
        interest_intersection = set(p_id_interest) & set(new_id_interest)
        new_p_interest_difference = set(new_id_interest) - set(p_id_interest)
        p_new_interest_difference = set(p_id_interest)  - set(new_id_interest)
        if len(Influencer_List) < influence_threshold:
            Influencer_List.insert(0,p_new_interest_difference)
        else:
            Influencer_List.pop()
            Influencer_List.insert(0,p_new_interest_difference)
            
        if len(Influencer_List) == influence_threshold:
            Influencer = Intersection_List(Influencer_List)
            Peoples[new_id][2] += list(Influencer)
            
        if len(interest_intersection) > same_interest_threshold or len(new_p_interest_difference) == 0 or len(p_new_interest_difference) == 0:
            ID_List.append((new_id,p_id))
            ID = Get_ID(ID_List)
            Add_Relation(Peoples,new_id,p_id) 
        



Relation = []
for i in range(num_person):
    for j in range(len(Peoples[i][1])):
        Relation.append((i,Peoples[i][1][j]))
        
        
Source = [i[0] for i in Relation]
Target =[i[1] for i in Relation]


G = nx.Graph()
G.add_edges_from(Relation)
#nx.draw(G,pos = nx.spring_layout(G),node_size = 10,node_color='r',width=0.3)
G_UnDi = G.to_undirected()
r = nx.degree_assortativity_coefficient(G_UnDi)
#plt.title('Person: '+str(num_person) +'  assortativity: '+str(r))
#rl.append(r)

A = A = nx.clustering(G_UnDi)
C = np.mean(np.array(A.values()))
D = nx.degree_histogram(G_UnDi)
k = PowLawFit(D,0)



largest_cc = max(nx.connected_components(G_UnDi),key=len)
G_Connected = nx.subgraph(G_UnDi,largest_cc)
L = nx.average_shortest_path_length(G_Connected)


#B = Sort_Dict(G.degree())

#A = Sort_Dict(Counter(ID))


'''
print len([a[0] for a in Peoples if len(a[1])==1 ])*1.0/num_person

degree = [len(set(a[1])) for a in Peoples]

A = Historgam_List(degree,1)

k = PowLawFit(A,1)

nx.degree_assortativity_coefficient(G)

D = nx.degree_histogram(G_UnDi)

PowLawFit(D,1)

'''

#plt.plot(range(len(rl)),rl)
#plt.ylim([-1,1])