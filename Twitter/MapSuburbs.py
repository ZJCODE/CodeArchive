# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:08:57 2016

@author: ZJun
"""

import numpy as np
from suburb import suburbs_find,suburbs_get
import pandas as pd

def Get_Suburbs():
    
    suburbs1 = suburbs_get("./Data/SA1_2011_AUST/SA1_2011_AUST")
    suburbs2 = suburbs_get("./Data/SA2_2011_AUST/SA2_2011_AUST")
    suburbs3 = suburbs_get("./Data/SA3_2011_AUST/SA3_2011_AUST")
    suburbs = suburbs1
    suburbs.update(suburbs2)
    suburbs.update(suburbs3)
    del suburbs1
    del suburbs2
    del suburbs3
    return suburbs
    
  
#user_name = Data.usr[13][-9]
  
suburbs =  Get_Suburbs()
import time

def Map_Location(Data,suburbs):
    Test_Points_Tuple = zip(Data.latitude,Data.longitude)
    Suburbs = []
    District = []
    j=0
    for P in Test_Points_Tuple:
        try:
            t1 = time.time()
            i = suburbs_find(suburbs, P)
            Suburbs.append(suburbs["info"][i]["record"][3])
            District.append(suburbs["info"][i]["record"][-2])
            j = j+1
            t2 = time.time()
            print '====deal with ' + str(j) + 'th ==== Cost '+str(t2-t1)+' Seconds===='
            #print "{}: {}".format(i, suburbs["info"][i]["record"])
        except LookupError:
            Suburbs.append(np.nan)
            District.append(np.nan)
    
    Data['Location'] = Suburbs
    Data['District'] = District
    return Data
      

def MapPoint2Suburbs(Point):
    try:        
        i = suburbs_find(suburbs, Point)
        return suburbs["info"][i]["record"]
    except LookupError:
        return np.nan
      
if __name__ == '__main__':
    Data = pd.read_csv('Twitter_Pos.csv')
    PosData = Map_Location(Data,suburbs)
    PosData.to_csv('PosData.csv',index = False)
