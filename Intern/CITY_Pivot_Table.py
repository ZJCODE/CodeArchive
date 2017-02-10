# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 15:07:31 2016

@author: Admin
"""

import pandas as pd

Data = pd.read_csv('ALL_Digital.txt')

'''
len(Data)
Out[153]: 1432268
'''

def Chinese_Or_Not(W):
    import re
    try:
        temp = W.decode('utf8')    
        xx=u"([\u4e00-\u9fa5]+)"
        pattern = re.compile(xx)
        Result = re.findall(pattern,temp)
        if len(Result)>0:
            return True
        else:
            return False
    except:
            return False
            
Not_Chinese = [c for c in Data.CITY if not Chinese_Or_Not(c)]

Chinese_City = list(set(Data.CITY) - set(Not_Chinese))


def English_Or_Not(W):
    import re
    try:
        xx=r'[A-Za-z]'
        pattern = re.compile(xx)
        Result = re.findall(pattern,W)
        if len(Result)>0:
            return True
        else:
            return False
    except:
            return False
            
English = [c for c in Not_Chinese if English_Or_Not(c)]

'''
len(English)
Out[154]: 9023
'''

English_City = list(set(English))

'''
English_City

['JIAN SHI',
 'WANNING SHI',
 'ALASHANMENG',
 'QIONGZHONGLIZUMIAOZU ZIZHIXIANG',
 'UNDEFINED',
 'LINGSHUILIZU ZIZHIXIANG',
 'LEDONGLIZU ZIZHIXIANG',
 'JINZHOU SHI',
 'SHAN SHI',
 'PINGLIANG SHI',
 'WULANCHABU SHI',
 'ANSHUN SHI']
 
'''


English_Chinese_Tran_City=[
'吉安市',
'万宁市',
'阿拉善盟',
'琼中黎族苗族自治县',
'其他',
'陵水黎族自治县',
'乐东黎族自治县',
'锦州市',
'山市',
'平凉市',
'乌兰察布市',
'安顺市'
]


Original = Chinese_City + English_City
Translation = Chinese_City + English_Chinese_Tran_City

Dict_CITY = dict(zip(Original,Translation))

'''
len(Not_Chinese)-len(English)-len(Data[Data.CITY.isnull()])
Out[176]: 12744
'''

City_new = []

for city in Data.CITY:
    try:
        City_new.append(Dict_CITY[city])
    except:
        City_new.append('其他')

# See City List        
#pd.DataFrame(list(set(City_new)))


Data['City_new'] = City_new

Data_Nike = Data[Data.Channel == 'Nike.com']

Data_Tmall = Data[Data.Channel == 'TMALL']



Data_Nike_Lanuch = Data_Nike[Data_Nike.Product_Type == 'Launch']
Data_Nike_Not_Lanuch = Data_Nike[Data_Nike.Product_Type != 'Launch']



Nike_Launch_AMOUNT_CITY = Data_Nike_Lanuch.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')
Nike_NLaunch_AMOUNT_CITY = Data_Nike_Not_Lanuch.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')
#Nike_QUANTITY_CITY = Data_Nike.pivot_table('QUANTITY','Date','City_new',aggfunc='sum')

Tmall_AMOUNT_CITY = Data_Tmall.pivot_table('AMOUNT','Date','City_new',aggfunc='sum')
#Tmall_QUANTITY_CITY = Data_Tmall.pivot_table('QUANTITY','Date','City_new',aggfunc='sum')

#----Save File-------------------------------------------------------------------------
'''

Nike_AMOUNT_CITY.to_csv('Nike_AMOUNT_CITY.csv',sep = ',',encoding='gb2312',header=True,index = True)
Nike_QUANTITY_CITY.to_csv('Nike_QUANTITY_CITY.csv',sep = ',',encoding='gb2312',header=True,index = True)

Tmall_AMOUNT_CITY.to_csv('Tmall_AMOUNT_CITY.csv',sep = ',',encoding='gb2312',header=True,index = True)
Tmall_QUANTITY_CITY.to_csv('Tmall_QUANTITY_CITY.csv',sep = ',',encoding='gb2312',header=True,index = True)

'''


'''
#--------------------------------------------------------------------------
Nike_AMOUNT_CITY_SUM = Data_Nike.pivot_table('AMOUNT','City_new',aggfunc='sum')
Nike_QUANTITY_CITY_SUM = Data_Nike.pivot_table('QUANTITY','City_new',aggfunc='sum')

Tmall_AMOUNT_CITY_SUM = Data_Tmall.pivot_table('AMOUNT','City_new',aggfunc='sum')
Tmall_QUANTITY_CITY_SUM = Data_Tmall.pivot_table('QUANTITY','City_new',aggfunc='sum')
'''

#-------Save File----------------------------------------------------------------------
'''
Nike_AMOUNT_CITY_SUM.to_csv('Nike_AMOUNT_CITY_SUM.csv',sep = ',',encoding='gb2312',header=True,index = True)
Nike_QUANTITY_CITY_SUM.to_csv('Nike_QUANTITY_CITY_SUM.csv',sep = ',',encoding='gb2312',header=True,index = True)

Tmall_AMOUNT_CITY_SUM.to_csv('Tmall_AMOUNT_CITY_SUM.csv',sep = ',',encoding='gb2312',header=True,index = True)
Tmall_QUANTITY_CITY_SUM.to_csv('Tmall_QUANTITY_CITY_SUM.csv',sep = ',',encoding='gb2312',header=True,index = True)
'''
#--------------------------------------------------------------------------------------------
'''
Top = 51


Nike_City = pd.DataFrame(Nike_AMOUNT_CITY_SUM)
Tmall_City = pd.DataFrame(Tmall_AMOUNT_CITY_SUM)

Nike_City_Sort = Nike_City.sort(columns = 'AMOUNT',ascending = False)
Tmall_City_Sort = Tmall_City.sort(columns = 'AMOUNT',ascending = False)

Nike_City_Top = list(Nike_City_Sort[:Top].index)
Nike_City_Top.remove('其他')

Tmall_City_Top = list(Tmall_City_Sort[:Top].index)
Tmall_City_Top.remove('其他')

Nike_AMOUNT_CITY_TOP = Nike_AMOUNT_CITY[Nike_City_Top]
Nike_QUANTITY_CITY_TOP = Nike_QUANTITY_CITY[Nike_City_Top]

Tmall_AMOUNT_CITY_TOP =Tmall_AMOUNT_CITY[Tmall_City_Top]
Tmall_QUANTITY_CITY_TOP = Tmall_QUANTITY_CITY[Tmall_City_Top]

Nike_AMOUNT_CITY_TOP.to_csv('Nike_AMOUNT_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)
Nike_QUANTITY_CITY_TOP.to_csv('Nike_QUANTITY_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)

Tmall_AMOUNT_CITY_TOP.to_csv('Tmall_AMOUNT_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)
Tmall_QUANTITY_CITY_TOP.to_csv('Tmall_QUANTITY_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)



'''



# Top 50

Top_City_Raw = pd.read_csv('Top50.txt')
City_Top = [c[0]+'市' for c in Top_City_Raw.values if Chinese_Or_Not(c[0])]

#----------------------------------------------------------------

Nike_Launch_AMOUNT_CITY_TOP = Nike_Launch_AMOUNT_CITY[City_Top]
Nike_NLaunch_AMOUNT_CITY_TOP = Nike_NLaunch_AMOUNT_CITY[City_Top]

#Nike_QUANTITY_CITY_TOP = Nike_QUANTITY_CITY[City_Top]
outlier = '\xe4\xbd\x9b\xe5\xb1\xb1\xe5\xb8\x82'
City_Top.remove(outlier)
Tmall_AMOUNT_CITY_TOP =Tmall_AMOUNT_CITY[City_Top]
#Tmall_QUANTITY_CITY_TOP = Tmall_QUANTITY_CITY[City_Top]

#--------------------------------------------------------------------------

Nike_Launch_AMOUNT_CITY_TOP.to_csv('.//City//Nike_Launch_AMOUNT_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)
Nike_NLaunch_AMOUNT_CITY_TOP .to_csv('.//City//Nike_NLaunch_AMOUNT_CITY_TOP .csv',sep = ',',encoding='gb2312',header=True,index = True)
#Nike_QUANTITY_CITY_TOP.to_csv('Nike_QUANTITY_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)

Tmall_AMOUNT_CITY_TOP.to_csv('.//City//Tmall_AMOUNT_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)
#Tmall_QUANTITY_CITY_TOP.to_csv('Tmall_QUANTITY_CITY_TOP.csv',sep = ',',encoding='gb2312',header=True,index = True)



#--------------------------------------------------------------------------------------

Top_City_Raw = pd.read_csv('Top50.txt')
City_Top = [c[0] for c in Top_City_Raw.values if Chinese_Or_Not(c[0])]


NSO_ALL_Week = pd.read_excel('.//Offline//Important//NSO_ALL_Week.xlsx')
NSO_Daily = pd.read_excel('.//Offline//Important//NSO_Daily.xlsx')
NFS_Daily = pd.read_excel('.//Offline//Important//NFS_Daily.xlsx')
NFS_ALL_Week= pd.read_excel('.//Offline//Important//NFS_ALL_Week.xlsx')

NSO_ALL_Week_City_AMOUNT = NSO_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','City',aggfunc = 'sum')
NFS_ALL_Week_City_AMOUNT = NFS_ALL_Week.pivot_table('SLS_USD','WEEK_DESCRIPTION','City',aggfunc = 'sum')
NSO_Daily_City_Amount = NSO_Daily.pivot_table('SLS_USD','TRAN_DT','City',aggfunc = 'sum')
NFS_Daily_City_Amount = NFS_Daily.pivot_table('SLS_USD','TRAN_DT','City',aggfunc = 'sum')


City_Top = [unicode(c,'utf-8') for c in City_Top]

City_T = [c for c in City_Top if c in NSO_ALL_Week_City_AMOUNT.columns]
NSO_ALL_Week_City_AMOUNT = NSO_ALL_Week_City_AMOUNT[City_T]

City_T = [c for c in City_Top if c in NFS_ALL_Week_City_AMOUNT.columns]
NFS_ALL_Week_City_AMOUNT = NFS_ALL_Week_City_AMOUNT[City_T]

City_T = [c for c in City_Top if c in NSO_Daily_City_Amount.columns]
NSO_Daily_City_Amount = NSO_Daily_City_Amount[City_T]

City_T = [c for c in City_Top if c in NFS_Daily_City_Amount.columns]
NFS_Daily_City_Amount = NFS_Daily_City_Amount[City_T]



NSO_ALL_Week_City_AMOUNT.to_csv('.//Offline//Important//NSO_ALL_Week_City_AMOUNT.csv',sep = ',',encoding='gb2312',header=True,index = True)
NFS_ALL_Week_City_AMOUNT.to_csv('.//Offline//Important//NFS_ALL_Week_City_AMOUNT.csv',sep = ',',encoding='gb2312',header=True,index = True)
NSO_Daily_City_Amount.to_csv('.//Offline//Important//NSO_Daily_City_Amount.csv',sep = ',',encoding='gb2312',header=True,index = True)
NFS_Daily_City_Amount.to_csv('.//Offline//Important//NFS_Daily_City_Amount.csv',sep = ',',encoding='gb2312',header=True,index = True)

