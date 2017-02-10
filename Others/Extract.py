# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:39:37 2016

@author: ZJun
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Files = glob.glob("*.xlsx")

# Replace A,B,C etc. to columns name

A = []
B = []
C = []
D = []
E = []
F = []
G = []
H = []
I = []
J = []
K = []
City = []
Year = []

# Construct a Dictionary to Store List for the convenience of iterating
All_Dict = dict(zip(range(1,12),[A,B,C,D,E,F,G,H,I,J,K])) 

for f in Files:
	# RawDataSheet is a Dictionary for sheet's name and data in that sheet     
    RawDataSheet = pd.read_excel(f,sheetname=None)
    for city in RawDataSheet.keys():
        RawData = RawDataSheet[city]  # Get Raw Data For Specific City
        for i in range(1,12):
            index = 4*(i-1)  # The Index For Specific Data 
            All_Dict[i] += list(RawData.ix[index:index+2,1]) # Extract Specific Data
        City += [city]*3 # Add City Name
        Year += [2013,2014,2015] # Add Year
        
Data = pd.DataFrame({'City':City,'A':A,'B':B,'C':C,'D':D,'E':E,'F':F,'G':G,'H':H,'I':I,'J':J,'K':K,'Year':Year})
Data= Data[['City','A','B','C','D','E','F','G','H','I','J','K','Year']]
            
Data.to_csv('ExtractData.csv',encoding='utf8',header=True,index = False)
        
        



