# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 14:34:14 2016

@author: Admin
"""

import pandas as pd

'''
Data = pd.read_csv('ALL_Digital.txt')

#----For Digital ------------------

Gender_Dict = {
 'BOYS TODDLER':'K',
 'GIRL PRE SCHOOL':'K',
 'LITTLE GIRLS':'K',
 'GIRLS TODDLER':'K',
 'GIRL GRADE SCHL':'K',
 'INFANT UNISEX':'K',
 'YOUNG WOMEN':'W',
 'BOYS GRADE SCHL':'K',
 'GRD SCHOOL UNSX':'K',
 'BOYS INFANT':'K',
 'GIRLS':'K',
 'LITTLE BOYS':'K',
 'GIRLS INFANT':'K',
 'BOYS PRE SCHOOL':'K',
 'YOUTH UNISEX':'K',
 'MENS':'M',
 'ADULT UNISEX':'U',
 'CHILD UNISEX':'K',
 'TODDLER UNISEX':'K',
 'BOYS':'K',
 'WOMENS':'W',
 'PRE SCHOOL UNSX':'K'}

#----------------------

for G in Data.GENDER:
    try :
         Gender_new.append(Gender_Dict[G])
    except:
        Gender_new.append(None)
        
Data['Gender_New'] = Gender_new

Gender_Add_DIVISION = []

for G,T in zip(Data.Gender_New,Data.DIVISION):
    if G == 'W' or G == 'M':
        Gender_Add_DIVISION.append(G+T.lower())
    else:
        Gender_Add_DIVISION.append(G)
		
Data['Gender_Add_DIVISION'] = Gender_Add_DIVISION

    
Data_Nike = Data[Data.Channel == 'Nike.com']

Data_Tmall = Data[Data.Channel == 'TMALL']

Data_Nike_Lanuch = Data_Nike[Data_Nike.Product_Type == 'Launch']
Data_Nike_Not_Lanuch = Data_Nike[Data_Nike.Product_Type != 'Launch']


Data_Nike_Lanuch.to_csv('Data_Nike_Launch.csv',sep = ',',encoding='gb2312',header=True,index = False)
Data_Nike_Not_Lanuch.to_csv('Data_Nike_Not_Lanuch.csv',sep = ',',encoding='gb2312',header=True,index = False)
Data_Tmall.to_csv('Data_Tmall.txt',sep = ',',encoding='utf8',header=True,index = False)
'''

#--------------------------------------------------------------


#----------------------------------------------------------------------------------------------

#--------2016.6.17----Online--Territory & Tier ----------------------

import pandas as pd

Data_Nike_Lanuch = pd.read_csv('.//Nike_Online\\Data_Nike_Launch.csv',sep = ',',encoding='gb2312')
Data_Nike_Not_Lanuch = pd.read_csv('.//Nike_Online\\Data_Nike_Not_Lanuch.csv',sep = ',',encoding='gb2312')
Data_Tmall = pd.read_csv('.//Nike_Online\\Data_Tmall.txt')

Data_Tmall_City_new = [unicode(c,'utf8') for c in Data_Tmall.City_new]
Data_Tmall['City_new'] = Data_Tmall_City_new

Online_City_Top = pd.read_excel('.//Nike_Online\\Online_City_Top.xlsx')

#---------------------------------------------
City_Territory_Dict = dict(zip(Online_City_Top.City_Top.values,Online_City_Top.Territory.values))

Nike_Launch_Territory = []
Nike_Not_Launch_Territory = []
Tmall_Territory = []

for c in Data_Nike_Lanuch.City_new:
    try:
        territory = City_Territory_Dict[c]
        Nike_Launch_Territory.append(territory)
    except:
        Nike_Launch_Territory.append('No_Territory_Info')

Data_Nike_Lanuch['Territory'] = Nike_Launch_Territory
        
for c in Data_Nike_Not_Lanuch.City_new:
    try:
        territory = City_Territory_Dict[c]
        Nike_Not_Launch_Territory.append(territory)
    except:
        Nike_Not_Launch_Territory.append('No_Territory_Info')
        
Data_Nike_Not_Lanuch['Territory'] = Nike_Not_Launch_Territory

for c in Data_Tmall.City_new:
    try:
        territory = City_Territory_Dict[c]
        Tmall_Territory.append(territory)
    except:
        Tmall_Territory.append('No_Territory_Info')

Data_Tmall['Territory'] = Tmall_Territory

#------------------------------------------------------------------------------------
City_Tier_Dict = dict(zip(Online_City_Top.City_Top.values,Online_City_Top.City_Tier.values))

Nike_Launch_Tier = []
Nike_Not_Launch_Tier = []
Tmall_Tier = []

for c in Data_Nike_Lanuch.City_new:
    try:
        tier = City_Tier_Dict[c]
        Nike_Launch_Tier.append(tier)
    except:
        Nike_Launch_Tier.append('No_Tier_Info')

Data_Nike_Lanuch['Tier'] = Nike_Launch_Tier
        
for c in Data_Nike_Not_Lanuch.City_new:
    try:
        tier = City_Tier_Dict[c]
        Nike_Not_Launch_Tier.append(tier)
    except:
        Nike_Not_Launch_Tier.append('No_Tier_Info')
        
Data_Nike_Not_Lanuch['Tier'] = Nike_Not_Launch_Tier

for c in Data_Tmall.City_new:
    try:
        tier = City_Tier_Dict[c]
        Tmall_Tier.append(tier)
    except:
        Tmall_Tier.append('No_Tier_Info')

Data_Tmall['Tier'] = Tmall_Tier

Nike_Lanuch_Territory = Data_Nike_Lanuch.pivot_table('AMOUNT','Date','Territory',aggfunc = 'sum')
Nike_Not_Lanuch_Territory = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date','Territory',aggfunc = 'sum')
Tmall_Territory = Data_Tmall.pivot_table('AMOUNT','Date','Territory',aggfunc = 'sum')

Nike_Lanuch_Territory.to_csv('Nike_Lanuch_Territory.csv')
Nike_Not_Lanuch_Territory.to_csv('Nike_Not_Lanuch_Territory.csv')
Tmall_Territory.to_csv('Tmall_Territory.csv')



Nike_Lanuch_Tier = Data_Nike_Lanuch.pivot_table('AMOUNT','Date','Tier',aggfunc = 'sum')
Nike_Not_Lanuch_Tier = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date','Tier',aggfunc = 'sum')
Tmall_Tier = Data_Tmall.pivot_table('AMOUNT','Date','Tier',aggfunc = 'sum')

Nike_Lanuch_Tier.to_csv('Nike_Lanuch_Tier.csv')
Nike_Not_Lanuch_Tier.to_csv('Nike_Not_Lanuch_Tier.csv')
Tmall_Tier.to_csv('Tmall_Tier.csv')


#-------Online----Category --Amount---------------------------------
Data_Nike_Lanuch_Category = Data_Nike_Lanuch.pivot_table('AMOUNT','Date','Gender_Add_DIVISION',aggfunc = 'sum')
Data_Nike_Lanuch_Category.to_csv('Data_Nike_Lanuch_Category.csv')
Data_Nike_Not_Lanuch_Category = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date','Gender_Add_DIVISION',aggfunc = 'sum')
Data_Nike_Not_Lanuch_Category.to_csv('Data_Nike_Not_Lanuch_Category.csv')
Data_Tmall_Category = Data_Tmall.pivot_table('AMOUNT','Date','Gender_Add_DIVISION',aggfunc = 'sum')
Data_Tmall_Category.to_csv('Data_Tmall_Category.csv')

Data_Nike_Lanuch_Amount = Data_Nike_Lanuch.pivot_table('AMOUNT','Date',aggfunc = 'sum')
Data_Nike_Lanuch_Amount.to_csv('Data_Nike_Lanuch_Amount.csv')
Data_Nike_Not_Lanuch_Amount = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date',aggfunc = 'sum')
Data_Nike_Not_Lanuch_Amount.to_csv('Data_Nike_Not_Lanuch_Amount.csv')
Data_Tmall_Amount = Data_Tmall.pivot_table('AMOUNT','Date',aggfunc = 'sum')
Data_Tmall_Amount.to_csv('Data_Tmall_Amount.csv')

#---Online ----CIty----------------------------------------------------------------------

Online_City_Top = pd.read_excel('.//Nike_Online\\Online_City_Top.xlsx')

City_Top = Online_City_Top.City_Top

Nike_Launch_AMOUNT_CITY = Data_Nike_Lanuch.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')
Nike_NLaunch_AMOUNT_CITY = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')
Tmall_AMOUNT_CITY = Data_Tmall.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')


City_T = [c for c in City_Top if c in Nike_Launch_AMOUNT_CITY.columns]
Nike_Launch_AMOUNT_CITY_Top = Nike_Launch_AMOUNT_CITY[City_T]
City_T = [c for c in City_Top if c in Nike_NLaunch_AMOUNT_CITY.columns]
Nike_NLaunch_AMOUNT_CITY_Top = Nike_Launch_AMOUNT_CITY[City_T]
City_T = [c for c in City_Top if c in Tmall_AMOUNT_CITY.columns]
Tmall_AMOUNT_CITY_Top = Nike_Launch_AMOUNT_CITY[City_T]

Nike_Launch_AMOUNT_CITY_Top.to_csv('Nike_Launch_AMOUNT_CITY_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)
Nike_NLaunch_AMOUNT_CITY_Top.to_csv('Nike_NLaunch_AMOUNT_CITY_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)
Tmall_AMOUNT_CITY_Top.to_csv('Tmall_AMOUNT_CITY_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)


#------2016.6.17---NSO Week---Territory--Tier-----Category--Type--City--------------------------------------------------------

NSO_ALL_Week = pd.read_excel('.//Nike_Offline\\NSO_ALL_Week.xlsx',sep = ',',encoding='gb2312')
Offline_City_Top = pd.read_excel('.//Nike_Offline\\Offline_City_Top.xlsx')

Gender_Dict = {'KIDS':'K','MEN':'M','UNISEX':'U','WOMEN':'W'}

Gender_new = [] 

for G in NSO_ALL_Week['RETL_GNDR_GRP']:
    try :
         Gender_new.append(Gender_Dict[G])
    except:
        Gender_new.append(None)

NSO_ALL_Week['Gender_New'] = Gender_new

Gender_Add_DIVISION = []

for G,T in zip(NSO_ALL_Week.Gender_New,NSO_ALL_Week.DIVISION):
    if T == 'APP' or T == 'FTW':        
        if G == 'W' or G == 'M':
            Gender_Add_DIVISION.append(G+T)
        else:
            Gender_Add_DIVISION.append(G)
    else:
        Gender_Add_DIVISION.append(G)

NSO_ALL_Week['Gender_Add_DIVISION'] = Gender_Add_DIVISION


TW = u'\u53f0\u6e7e'
XG = u'\u9999\u6e2f'
NSO_ALL_Week = NSO_ALL_Week[NSO_ALL_Week.City != TW]
NSO_ALL_Week = NSO_ALL_Week[NSO_ALL_Week.City != XG]

NSO_ALL_Week_ES = NSO_ALL_Week[NSO_ALL_Week.STORE_TYPE == 'ES']
NSO_ALL_Week_Not_ES = NSO_ALL_Week[NSO_ALL_Week.STORE_TYPE != 'ES']

NSO_ALL_Week_Amount = NSO_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION',aggfunc='sum')
NSO_ALL_Week_Amount.to_csv('NSO_ALL_Week_Amount.csv',header=True,index = True)


NSO_ALL_Week_Not_ES_Amount = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION',aggfunc='sum')
NSO_ALL_Week_Not_ES_Amount.to_csv('NSO_ALL_Week_Not_ES_Amount.csv',header=True,index = True)

NSO_ALL_Week_ES_Amount = NSO_ALL_Week_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION',aggfunc='sum')
NSO_ALL_Week_ES_Amount.to_csv('NSO_ALL_Week_ES_Amount.csv',header=True,index = True)



#-------------------------------------------------------------------------

City_Tier_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.City_Tier.values))

NSO_ALL_Week_Not_ES_Tier = []

for c in NSO_ALL_Week_Not_ES.City:
    try:
        tier = City_Tier_Dict[c]
        NSO_ALL_Week_Not_ES_Tier.append(tier)
    except:
        NSO_ALL_Week_Not_ES_Tier.append('No_Tier_Info')
        
NSO_ALL_Week_Not_ES['Tier'] = NSO_ALL_Week_Not_ES_Tier


City_Territory_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.Territory.values))

NSO_ALL_Week_Not_ES_Territory = []

for c in NSO_ALL_Week_Not_ES.City:
    try:
        Territory = City_Territory_Dict[c]
        NSO_ALL_Week_Not_ES_Territory.append(Territory)
    except:
        NSO_ALL_Week_Not_ES_Territory.append('No_Territory_Info')
        
NSO_ALL_Week_Not_ES['Territory'] = NSO_ALL_Week_Not_ES_Territory

NSO_ALL_Week_Not_ES_Tier = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION','Tier',aggfunc = 'sum')
NSO_ALL_Week_Not_ES_Territory = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION','Territory',aggfunc = 'sum')
NSO_ALL_Week_Not_ES_Category = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION','Gender_Add_DIVISION',aggfunc = 'sum')
NSO_ALL_Week_Not_ES_Type = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION','STORE_TYPE',aggfunc = 'sum')

NSO_ALL_Week_Not_ES_Tier.to_csv('NSO_ALL_Week_Not_ES_Tier.csv')
NSO_ALL_Week_Not_ES_Territory.to_csv('NSO_ALL_Week_Not_ES_Territory.csv')
NSO_ALL_Week_Not_ES_Category.to_csv('NSO_ALL_Week_Not_ES_Category.csv')
NSO_ALL_Week_Not_ES_Type.to_csv('NSO_ALL_Week_Not_ES_Type.csv')

#------------------------------------------------------------------
City_Top = Offline_City_Top.City_Top
NSO_ALL_Week_Not_ES_City = NSO_ALL_Week_Not_ES.pivot_table('SLS_USD','WEEK_DESCRIPTION','City',aggfunc = 'sum')
City_T = [c for c in City_Top if c in NSO_ALL_Week_Not_ES_City.columns]
NSO_ALL_Week_Not_ES_City_Top = NSO_ALL_Week_Not_ES_City[City_T]

NSO_ALL_Week_Not_ES_City_Top.to_csv('NSO_ALL_Week_Not_ES_City_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)


#---------------------------------------
#-----------NSO -Daily--Territory--Tier--Category--Type--City-------------------

NSO_Daily = pd.read_excel('.//Nike_Offline\\NSO_Daily.xlsx',sep = ',',encoding='gb2312')
Offline_City_Top = pd.read_excel('.//Nike_Offline\\Offline_City_Top.xlsx')

Gender_Dict = {'KIDS':'K','MEN':'M','UNISEX':'U','WOMEN':'W'}

Gender_new = [] 

for G in NSO_Daily['RETL_GNDR_GRP']:
    try :
         Gender_new.append(Gender_Dict[G])
    except:
        Gender_new.append(None)

NSO_Daily['Gender_New'] = Gender_new

Gender_Add_DIVISION = []

for G,T in zip(NSO_Daily.Gender_New,NSO_Daily.DIVISION):
    if T == 'APP' or T == 'FTW':        
        if G == 'W' or G == 'M':
            Gender_Add_DIVISION.append(G+T)
        else:
            Gender_Add_DIVISION.append(G)
    else:
        Gender_Add_DIVISION.append(G)

NSO_Daily['Gender_Add_DIVISION'] = Gender_Add_DIVISION


TW = u'\u53f0\u6e7e'
XG = u'\u9999\u6e2f'
NSO_Daily = NSO_Daily[NSO_Daily.City != TW]
NSO_Daily = NSO_Daily[NSO_Daily.City != XG]



NSO_Daily_ES = NSO_Daily[NSO_Daily.STORE_TYPE == 'ES']
NSO_Daily_Not_ES = NSO_Daily[NSO_Daily.STORE_TYPE != 'ES']

NSO_Daily_Amount = NSO_Daily.pivot_table('SLS_USD','TRAN_DT',aggfunc='sum')
NSO_Daily_Amount.to_csv('NSO_Daily_Amount.csv',header=True,index = True)


NSO_Daily_Not_ES_Amount = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT',aggfunc='sum')
NSO_Daily_Not_ES_Amount.to_csv('NSO_Daily_Not_ES_Amount.csv',header=True,index = True)

NSO_Daily_ES_Amount = NSO_Daily_ES.pivot_table('SLS_USD','TRAN_DT',aggfunc='sum')
NSO_Daily_ES_Amount.to_csv('NSO_Daily_ES_Amount.csv',header=True,index = True)



#-------------------------------------------------------------------------

City_Tier_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.City_Tier.values))

NSO_Daily_Not_ES_Tier = []

for c in NSO_Daily_Not_ES.City:
    try:
        tier = City_Tier_Dict[c]
        NSO_Daily_Not_ES_Tier.append(tier)
    except:
        NSO_Daily_Not_ES_Tier.append('No_Tier_Info')
        
NSO_Daily_Not_ES['Tier'] = NSO_Daily_Not_ES_Tier


City_Territory_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.Territory.values))

NSO_Daily_Not_ES_Territory = []

for c in NSO_Daily_Not_ES.City:
    try:
        Territory = City_Territory_Dict[c]
        NSO_Daily_Not_ES_Territory.append(Territory)
    except:
        NSO_Daily_Not_ES_Territory.append('No_Territory_Info')
        
NSO_Daily_Not_ES['Territory'] = NSO_Daily_Not_ES_Territory

NSO_Daily_Not_ES_Tier = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT','Tier',aggfunc = 'sum')
NSO_Daily_Not_ES_Territory = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT','Territory',aggfunc = 'sum')
NSO_Daily_Not_ES_Category = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT','Gender_Add_DIVISION',aggfunc = 'sum')
NSO_Daily_Not_ES_Type = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT','STORE_TYPE',aggfunc = 'sum')

NSO_Daily_Not_ES_Tier.to_csv('NSO_Daily_Not_ES_Tier.csv')
NSO_Daily_Not_ES_Territory.to_csv('NSO_Daily_Not_ES_Territory.csv')
NSO_Daily_Not_ES_Category.to_csv('NSO_Daily_Not_ES_Category.csv')
NSO_Daily_Not_ES_Type.to_csv('NSO_Daily_Not_ES_Type.csv')

#------------------------------------------------------------------
City_Top = Offline_City_Top.City_Top
NSO_Daily_Not_ES_City = NSO_Daily_Not_ES.pivot_table('SLS_USD','TRAN_DT','City',aggfunc = 'sum')
City_T = [c for c in City_Top if c in NSO_Daily_Not_ES_City.columns]
NSO_Daily_Not_ES_City_Top = NSO_Daily_Not_ES_City[City_T]

NSO_Daily_Not_ES_City_Top.to_csv('NSO_Daily_Not_ES_City_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)

#-------------------------------------------------------------------
#--------------NFS-----Weekly-----------------------------------------------

NFS_ALL_Week = pd.read_excel('.//Nike_Offline\\NFS_ALL_Week.xlsx',sep = ',',encoding='gb2312')




Offline_City_Top = pd.read_excel('.//Nike_Offline\\Offline_City_Top.xlsx')

Gender_Dict = {'KIDS':'K','MEN':'M','UNISEX':'U','WOMEN':'W'}

Gender_new = [] 

for G in NFS_ALL_Week['RETL_GNDR_GRP']:
    try :
         Gender_new.append(Gender_Dict[G])
    except:
        Gender_new.append(None)

NFS_ALL_Week['Gender_New'] = Gender_new

Gender_Add_DIVISION = []

for G,T in zip(NFS_ALL_Week.Gender_New,NFS_ALL_Week.DIVISION):
    if T == 'APP' or T == 'FTW':        
        if G == 'W' or G == 'M':
            Gender_Add_DIVISION.append(G+T)
        else:
            Gender_Add_DIVISION.append(G)
    else:
        Gender_Add_DIVISION.append(G)

NFS_ALL_Week['Gender_Add_DIVISION'] = Gender_Add_DIVISION

TW = u'\u53f0\u6e7e'
XG = u'\u9999\u6e2f'
NFS_ALL_Week = NFS_ALL_Week[NFS_ALL_Week.City != TW]
NFS_ALL_Week = NFS_ALL_Week[NFS_ALL_Week.City != XG]

NFS_ALL_Week_Amount = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION',aggfunc='sum')
NFS_ALL_Week_Amount.to_csv('NFS_ALL_Week_Amount.csv',header=True,index = True)


City_Tier_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.City_Tier.values))

NFS_ALL_Week_Tier = []

for c in NFS_ALL_Week.City:
    try:
        tier = City_Tier_Dict[c]
        NFS_ALL_Week_Tier.append(tier)
    except:
        NFS_ALL_Week_Tier.append('No_Tier_Info')
        
NFS_ALL_Week['Tier'] = NFS_ALL_Week_Tier


City_Territory_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.Territory.values))

NFS_ALL_Week_Territory = []

for c in NFS_ALL_Week.City:
    try:
        Territory = City_Territory_Dict[c]
        NFS_ALL_Week_Territory.append(Territory)
    except:
        NFS_ALL_Week_Territory.append('No_Territory_Info')
        
NFS_ALL_Week['Territory'] = NFS_ALL_Week_Territory

NFS_ALL_Week_Tier = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','Tier',aggfunc = 'sum')
NFS_ALL_Week_Territory = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','Territory',aggfunc = 'sum')
NFS_ALL_Week_Category = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','Gender_Add_DIVISION',aggfunc = 'sum')
NFS_ALL_Week_Type = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','STORE_TYPE',aggfunc = 'sum')

NFS_ALL_Week_Tier.to_csv('NFS_ALL_Week_Tier.csv')
NFS_ALL_Week_Territory.to_csv('NFS_ALL_Week_Territory.csv')
NFS_ALL_Week_Category.to_csv('NFS_ALL_Week_Category.csv')
NFS_ALL_Week_Type.to_csv('NFS_ALL_Week_Type.csv')

#------------------------------------------------------------------
City_Top = Offline_City_Top.City_Top
NFS_ALL_Week_Type_City = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','City',aggfunc = 'sum')
City_T = [c for c in City_Top if c in NFS_ALL_Week_Type_City.columns]
NFS_ALL_Week_Type_City_Top = NFS_ALL_Week_Type_City[City_T]

NFS_ALL_Week_Type_City_Top.to_csv('NFS_ALL_Week_Type_City_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)

#----------------------------------------------------------------
#--------------NFS-----Daily-----------------------------------------------


NFS_Daily = pd.read_excel('.//Nike_Offline\\NFS_Daily.xlsx',sep = ',',encoding='gb2312')




Offline_City_Top = pd.read_excel('.//Nike_Offline\\Offline_City_Top.xlsx')

Gender_Dict = {'KIDS':'K','MEN':'M','UNISEX':'U','WOMEN':'W'}

Gender_new = [] 

for G in NFS_Daily['RETL_GNDR_GRP']:
    try :
         Gender_new.append(Gender_Dict[G])
    except:
        Gender_new.append(None)

NFS_Daily['Gender_New'] = Gender_new

Gender_Add_DIVISION = []

for G,T in zip(NFS_Daily.Gender_New,NFS_Daily.DIVISION):
    if T == 'APP' or T == 'FTW':        
        if G == 'W' or G == 'M':
            Gender_Add_DIVISION.append(G+T)
        else:
            Gender_Add_DIVISION.append(G)
    else:
        Gender_Add_DIVISION.append(G)

NFS_Daily['Gender_Add_DIVISION'] = Gender_Add_DIVISION

TW = u'\u53f0\u6e7e'
XG = u'\u9999\u6e2f'
NFS_Daily = NFS_Daily[NFS_Daily.City != TW]
NFS_Daily = NFS_Daily[NFS_Daily.City != XG]

NFS_Daily_Amount = NFS_Daily.pivot_table('SLS_USD','TRAN_DT',aggfunc='sum')
NFS_Daily_Amount.to_csv('NFS_Daily_Amount.csv',header=True,index = True)


City_Tier_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.City_Tier.values))

NFS_Daily_Tier = []

for c in NFS_Daily.City:
    try:
        tier = City_Tier_Dict[c]
        NFS_Daily_Tier.append(tier)
    except:
        NFS_Daily_Tier.append('No_Tier_Info')
        
NFS_Daily['Tier'] = NFS_Daily_Tier


City_Territory_Dict = dict(zip(Offline_City_Top.City_Top.values,Offline_City_Top.Territory.values))

NFS_Daily_Territory = []

for c in NFS_Daily.City:
    try:
        Territory = City_Territory_Dict[c]
        NFS_Daily_Territory.append(Territory)
    except:
        NFS_Daily_Territory.append('No_Territory_Info')
        
NFS_Daily['Territory'] = NFS_Daily_Territory

NFS_Daily_Tier = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','Tier',aggfunc = 'sum')
NFS_Daily_Territory = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','Territory',aggfunc = 'sum')
NFS_Daily_Category = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','Gender_Add_DIVISION',aggfunc = 'sum')
NFS_Daily_Type = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','STORE_TYPE',aggfunc = 'sum')

NFS_Daily_Tier.to_csv('NFS_Daily_Tier.csv')
NFS_Daily_Territory.to_csv('NFS_Daily_Territory.csv')
NFS_Daily_Category.to_csv('NFS_Daily_Category.csv')
NFS_Daily_Type.to_csv('NFS_Daily_Type.csv')

#------------------------------------------------------------------
City_Top = Offline_City_Top.City_Top
NFS_Daily_City = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','City',aggfunc = 'sum')
City_T = [c for c in City_Top if c in NFS_Daily_City.columns]
NFS_Daily_City_Top = NFS_Daily_City[City_T]

NFS_Daily_City_Top.to_csv('NFS_Daily_City_Top.csv',sep = ',',encoding='gb2312',header=True,index = True)
