# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:28:43 2016

@author: Admin
"""
import pandas as pd

def Save_List(List,Name):
    File = Name + '.txt'
    pd.DataFrame({Name:List}).to_csv(File,encoding='utf8',header=True,index = False)


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
    
def Import_Obj(File):    
    import pickle
    File_Name = File+'.pkl'
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)
    
def Reverse_Dict(Diction):  # 适用于一一映射的字典
    Items = Diction.items()
    key = [x[0] for x in Items]
    value = [x[1] for x in Items]
    Re_Dict = dict(zip(value,key))
    return Re_Dict
    
def Save_DataFrame(DF,File_Name):
    File = File_Name + '.txt'
    DF.to_csv(File,encoding='utf8',header=True,index = False)
    
def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)
    
def Get_ID_Board(Relation):
    import numpy as np
    Relation = Relation.dropna()
    Board_Name = list(set(Relation.B))
    Board_Name_Dict = dict(zip(Board_Name,range(len(Board_Name))))
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
    
        
    
    Index_Board = Reverse_Dict(Board_Name_Dict)
    
    Max_Index = M.argmax(1)
    
    Class_Board = [Index_Board[Max_Index[i,0]] for i in range(len(Max_Index))]
    
    ID_Class_Board = dict(zip(User,Class_Board))
    
    return ID_Class_Board