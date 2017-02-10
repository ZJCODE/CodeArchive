# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:08:25 2016

@author: Admin
"""


import re
import pandas as pd


def Get_Reply_Relation(Data):
    Reply_Relation = Data[['Post_ID','ID','Post_Position','Reply_Time']]
    Reply_Relation.columns = ['P','R','B','Time']
    return Reply_Relation
    

def Get_Inside_Relation(Data):
    patten = '\xe5\xbc\x95\xe7\x94\xa8\s(.*)\s\xe5\x8f\x91\xe8\xa1\xa8\xe4\xba\x8e'    
    Inside_Reply_People = []    
    Users = set(Data.Post_ID) | set(Data.ID)
    
    for content in Data.Content:
        try:
            people = re.findall(patten,content)[0]
        except:
            people = None
        if people in Users:
            Inside_Reply_People.append(people)
        else:
            Inside_Reply_People.append(None)
    
    Data['Inside_Reply_People'] = Inside_Reply_People
    Inside = Data[~ Data.Inside_Reply_People.isnull()]
    Inside_Relation = Inside[['ID','Inside_Reply_People','Post_Position','Reply_Time']]
    Inside_Relation.columns = ['P','R','B','Time']
    return Inside_Relation

def Get_Relation(Data,WithBoard=0,WithTime=0):
    Reply_Relation = Get_Reply_Relation(Data)
    Inside_Relation = Get_Inside_Relation(Data)
    Relation = pd.concat([Reply_Relation,Inside_Relation])
    if WithBoard == 0 and WithTime == 0: 
        return Relation[['P','R']]
    elif WithBoard == 1 and WithTime == 0: 
        return Relation[['P','R','B']]
    elif WithBoard == 0 and WithTime == 1: 
        return Relation[['P','R','Time']]

    
    