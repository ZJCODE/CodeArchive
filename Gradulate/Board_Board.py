# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:33:19 2016

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
from Tools import *

Data = Get_Data('2006/4','2012/3')

Relation = Get_Relation(Data,1,0)
Relation = Relation.dropna()

Board_Name = list(set(Relation.B))

Board_Name_Dict = dict(zip(Board_Name,range(len(Board_Name))))
Save_Obj(Board_Name_Dict,'Board_Name_Dict')

User = list(set(Relation.P)|set(Relation.R))

User_Dict = dict(zip(User,range(len(User))))

# People In Board 
M = np.matrix([0]*len(User)*len(Board_Name)).reshape(len(User),len(Board_Name))


    
for Id,board in zip(Relation.R,Relation.B):
    M[User_Dict[Id],Board_Name_Dict[board]] += 1
    
PB = Relation[['P','B']]
PBD = PB.drop_duplicates()


for Id,board in zip(PBD.P,PBD.B):
    M[User_Dict[Id],Board_Name_Dict[board]] += 1    


BB = np.matrix([0]*len(Board_Name)*len(Board_Name)).reshape(len(Board_Name),len(Board_Name))
'''
for i in range(len(M)):
    for j in range(len(Board_Name)):
        for k in range(j,len(Board_Name)):
            BB[j,k] += min(M[i,j],M[i,k])
            
Save_Obj(BB,'BB')
'''
BB_Relation = []

for i in range(len(M)):
    for j in range(len(Board_Name)):
        for k in range(j,len(Board_Name)):
            BB[j,k] += min(M[i,j],M[i,k])
            
Save_Obj(BB,'BB')

for i in range(len(Board_Name)):
    for j in range(i+1,len(Board_Name)):
        for k in range(BB[i,j]/100):
            BB_Relation.append([i,j])
            
BB_DF = pd.DataFrame(BB_Relation,columns=['Source','Target'])
            
Save_DataFrame(BB_DF,'BB_modify')

plt.plot(P2_DL)
plt.plot(P1_DL)
plt.ylim([0,10000])
#plt.xlim([0,1000])

        