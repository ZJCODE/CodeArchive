# -*- coding: utf-8 -*-
"""
Created on Wed May 04 21:53:31 2016

@author: Admin
"""

from Get_Relation import *
from Get_Data import *
from Time_Tool import *
from PowLaw import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import networkx as nx


Data = Get_Data('2006/4','2006/5')

Relation = Get_Relation(Data,0,1)

GTime = [Generailze_Time(time) for time in Relation.Time]

Relation['GTime'] = GTime

Sort_Relation = Relation.sort('GTime')

R = Sort_Relation[['P','R']]

#Number = input('N: ')
rl=[]
for Number in range(3,3000):
    
    G = nx.Graph()
    G.add_edges_from(R[:Number].values)
    
    #nx.draw(G,pos = nx.spring_layout(G),node_size = 10,node_color='r',width=0.3)
    r = nx.degree_assortativity_coefficient(G)
    #plt.title('Link: '+str(Number) +'  assortativity: '+str(r))
    rl.append(r)
    

plt.plot(rl)
plt.ylabel('assortativity')
G_UnDi = G.to_undirected()
degree = nx.degree_histogram(G_UnDi)
Gamma = PowLawFit(degree[1:],Draw = 1)

#plt.rc('figure', figsize=(10, 10))
