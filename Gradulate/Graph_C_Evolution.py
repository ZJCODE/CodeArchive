

import glob
import re
import pandas as pd
import networkx as nx
import numpy as np
from PowLaw import *
from Get_Relation import *
from Get_Data import *
from Time_Tool import *
#from Preferential import *



#-------------------------List Files Name------------------
Files = glob.glob('D:\BBS\BBS_Data\Year_Month_Reply\*.txt')
pattern = r'D:\\BBS\\BBS_Data\\Year_Month_Reply\\(\d+)'
Files = sorted(Files , key = lambda x : int(re.findall(pattern , x)[0]))
#----------------------------------------------------------



#-------------------Time List--------------------
Time = []
for year in range(2006,2017):
    for month in range(1,13):
        Time.append(str(year) + '/' + str(month))
Time_columns = Time[3:-10]
Time_Index_Dict = dict(zip(Time_columns,range(len(Time_columns))))
#-----------------------------------------------



#--------------------------Get Relation---------------------------------------------

ALL_Relation = pd.DataFrame(columns = ['P','R','B'])


Start = input('Start Time (year/month):    ')
End = input('Start Time (year/month):    ')

r=[]
k_mean = []
k_Max = []
Num_Node = []
Gamma_List = []
#Preferential = []
Cluster = []
Average_Path = []

for f in Files[Time_Index_Dict[Start]:Time_Index_Dict[End]+1]:
    Data = pd.read_csv(f)
    Reply_Relation = Get_Reply_Relation(Data)
    Inside_Relation = Get_Inside_Relation(Data)
    ALL_Relation = pd.concat([ALL_Relation,Reply_Relation,Inside_Relation])
    ALL_Relation = ALL_Relation.dropna()
    del Data
    #Relation_With_Time = ALL_Relation[['P','R','Time']]
    #Preferential.append(Preferential_Attachment(Relation_With_Time))
    Relation = ALL_Relation[['P','R']]
    G = nx.DiGraph()
    G.add_edges_from(Relation.values)
    G_UnDi = G.to_undirected()
    
    largest_cc = max(nx.connected_components(G_UnDi),key=len)
    G_Connected = nx.subgraph(G_UnDi,largest_cc)
    L = nx.average_shortest_path_length(G_Connected)
    Average_Path.append(L)
    
    
    Cluster.append(np.mean(np.array(nx.clustering(G_UnDi).values())))
    
    #r.append(nx.degree_assortativity_coefficient(G_UnDi))
    
    #Degree_List = G_UnDi.degree().values()
    #k_mean.append(np.mean(Degree_List))

        
    
    #k_Max.append(max(Degree_List))
    
    #degree = nx.degree_histogram(G_UnDi)
    #Gamma = PowLawFit(degree[1:])
    #Gamma_List.append(Gamma)    

    #Node = len(G_UnDi.nodes())
    #Num_Node.append(Node)



T = Time_columns[:Time_columns.index(End)+1]
Graph_CC = pd.DataFrame({'Time':T,'Average_Path':Average_Path,'Cluster':Cluster})
#Graph_C = pd.DataFrame({'Time':T,'r':r,'k_Mean':k_mean,'k_Max':k_Max,'Num_Node':Num_Node,'Gamma_List':Gamma_List,'Preference':Preferential})
#Graph_C.to_csv('Graph_C.txt',encoding='utf8',header=True,index = False)
#Graph_C.to_excel('Graph_C.xlsx',encoding='utf8',header=True,index = False)
Graph_C.to_excel('Graph_CC.xlsx',encoding='utf8',header=True,index = False)
