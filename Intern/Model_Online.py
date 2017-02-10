# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:30:31 2016

@author: Admin
"""

import pandas as pd
from datetime import datetime
import numpy as np





'''
Start_Time = datetime(2015,3,1)
End_Time = datetime(2016,3,31)

TS0 = pd.Series(Data.AMOUNT.values,index = Data.Date)
TS = TS[Start_Time:End_Time]
'''

def Set_None_Zero(L):
    List = []
    for i in L:
        if np.isnan(i):
            List.append(0)
        else:
            List.append(i)
    return List


def Get_BaseLine_NormalLine(Data_Input):
    
    Data = pd.DataFrame(Data_Input.values,columns=['Date','AMOUNT'])
    Data['AMOUNT'] = Set_None_Zero(Data.AMOUNT)
    Data['AMOUNT'] = Data['AMOUNT'].astype(int)
    TS = pd.Series(Data.AMOUNT.values,index = Data.Date)
#    Start_Time = datetime(2015,3,1)
#    End_Time = datetime(2016,3,31)
#    TS = TS[Start_Time:End_Time]  
    
    #--------------------------------------------------
    
#    Data = pd.DataFrame({'Date':TS.index,'AMOUNT':TS.values})
    Year_Month = []
    
    for d in TS.index:
        Year_Month.append(str(d.year)+'-'+str(d.month))
            
    Data['YM'] = Year_Month
    
    Average_YM = Data.pivot_table('AMOUNT','YM',aggfunc= 'mean')
    
    threshold = 0.1
    
    Average_YM_DF = pd.DataFrame({'YM':Average_YM.index,'Average':Average_YM.values})
    Low =[a*(1-threshold) for a in Average_YM_DF.Average]
    UP = [a*(1+threshold) for a in Average_YM_DF.Average]
    Average_YM_DF['UP'] = UP
    Average_YM_DF['Low'] = Low
    
    UP_Dict = dict(zip(Average_YM_DF.YM,Average_YM_DF.UP))
    Low_Dict = dict(zip(Average_YM_DF.YM,Average_YM_DF.Low))
    #-----------------------------------------------------------
    
    
    Base = list(Data.AMOUNT[:7])
    
    for i in range(7,len(TS[:datetime(2015,12,31)])):
        if Data.YM[i] == '2015-11' or Data.YM[i] == '2015-12':
            UP = UP_Dict['2015-10']
            Low = Low_Dict['2015-10']
        else:
            UP = UP_Dict[Data.YM[i]]
            Low = Low_Dict[Data.YM[i]]
        if Data.AMOUNT[i] < UP and Data.AMOUNT[i] > Low:
            Base.append(Data.AMOUNT[i])
        else:
            Base.append(np.mean(Base[i-7:i]))
            
    # Deal with 2016
            
    Base = Base + list(TS[datetime(2016,1,1):datetime(2016,1,7)].values)
    
    for i in range(len(TS[:datetime(2016,1,7)]),len(TS)):
        if Data.YM[i] == '2015-11' or Data.YM[i] == '2015-12':
            UP = UP_Dict['2015-10']
            Low = Low_Dict['2015-10']
        else:
            UP = UP_Dict[Data.YM[i]]
            Low = Low_Dict[Data.YM[i]]
        if Data.AMOUNT[i] < UP and Data.AMOUNT[i] > Low:
            Base.append(Data.AMOUNT[i])
        else:
            Base.append(np.mean(Base[i-7:i]))
    
    
    TS_Base = pd.Series(Base,index = Data.Date)
    
    from statsmodels.tsa.ar_model import AR
    ar_mod = AR(TS_Base)
    ARM = ar_mod.fit(7)
    AR_FIT = TS_Base[:7].append(ARM.fittedvalues)
    
    return TS_Base.index,TS.values,TS_Base.values,AR_FIT.values

'''
Result = pd.DataFrame({'Date':Data_ALL.Date})

for col in Columns[1:]:
    Data_Input = Data_ALL[[Columns[0],col]]
    t,s,b,n = Get_BaseLine_NormalLine(Data_Input)
    Result[col+'_Sales'] = s
    Result[col+'_BaseLine'] = b
    Result[col+'_NormalLine'] = n
    
Result.to_csv('Base_Normal_Nike_NLaunch.csv',sep = ',',encoding='utf8',header=True,index = False)
'''

'''
Result = pd.DataFrame({'Date':TS_Base.index,'Sales':TS.values,'BaseLine':TS_Base.values,'NormalLine':AR_FIT.values})


Sales_Incremental = Result.Sales - Result.NormalLine 
Sales_Uplift = Sales_Incremental / Result.NormalLine

Result['Sales_Incremental'] = Sales_Incremental
Result['Sales_Uplift'] = Sales_Uplift

Result.to_csv('Digital.csv',sep = ',',header=True,index = False)

plt.plot(TS.index,TS_Base)
plt.plot(AR_FIT.index,AR_FIT,'g')
'''



#-----------------------Nike.com   OutPut  ----------------------------------

Nike_L = pd.read_excel('Nike_Launch.xlsx',encoding='gb2312')

Data_ALL = pd.read_excel('Nike_NLaunch.xlsx',encoding='gb2312')


Columns = Data_ALL.columns

Sales = pd.DataFrame({'Date':Data_ALL.Date})
BaseLine = pd.DataFrame({'Date':Data_ALL.Date})
NormalLine = pd.DataFrame({'Date':Data_ALL.Date})

for col in Columns[1:]: 
    Data_Input = Data_ALL[[Columns[0],col]]
    t,s,b,n = Get_BaseLine_NormalLine(Data_Input)
    Sales[col] = s
    BaseLine[col] = b
    NormalLine[col] = n
    
for i in range(1,len(Sales.columns)):
    Data_Output = pd.DataFrame({'Date':Data_ALL.Date})
    Data_Output['Actual_Base'] = Sales.ix[:,i]
    Data_Output['BaseLine'] = BaseLine.ix[:,i]
    Data_Output['NormalLine'] = NormalLine.ix[:,i]
    Data_Output['Incremental'] = Data_Output['Actual_Base'] - Data_Output['NormalLine']
    Data_Output['Uplift'] = Data_Output['Incremental'] / Data_Output['NormalLine']
    Data_Output['Launch'] = Nike_L.ix[:,i]
    Data_Output['Actual_Total'] = Data_Output['Launch'] + Data_Output['Actual_Base']
    Data_Output['Normal_Total'] = Data_Output['Launch'] + Data_Output['NormalLine']
    Data_Output['Total_Incremental'] = Data_Output['Actual_Total'] - Data_Output['Normal_Total']
    Data_Output['Total_Uplift'] = Data_Output['Total_Incremental'] / Data_Output['Normal_Total']
    Data_Output.to_excel('.//City//Nike_'+Columns[i]+'.xlsx',encoding='utf8',header=True,index = False)
    
#--------------------------------------------------------------------------------------------
    
#-------------------------------Tmall OutPut---------------------------------------
    
Data_ALL = pd.read_excel('Tmall.xlsx')

Columns = Data_ALL.columns

Sales = pd.DataFrame({'Date':Data_ALL.Date})
BaseLine = pd.DataFrame({'Date':Data_ALL.Date})
NormalLine = pd.DataFrame({'Date':Data_ALL.Date})

for col in Columns[1:]: 
    Data_Input = Data_ALL[[Columns[0],col]]
    t,s,b,n = Get_BaseLine_NormalLine(Data_Input)
    Sales[col] = s
    BaseLine[col] = b
    NormalLine[col] = n
    
for i in range(1,len(Sales.columns)):
    Data_Output = pd.DataFrame({'Date':Data_ALL.Date})
    Data_Output['Total_Sales'] = Sales.ix[:,i]
    Data_Output['BaseLine'] = BaseLine.ix[:,i]
    Data_Output['NormalLine'] = NormalLine.ix[:,i]
    Data_Output['Incremental'] = Data_Output['Total_Sales'] - Data_Output['NormalLine']
    Data_Output['Uplift'] = Data_Output['Incremental'] / Data_Output['NormalLine']
    Data_Output.to_excel('.//City//Tmall_'+Columns[i]+'.xlsx',encoding='utf8',header=True,index = False)


#---------------------------------------------------------------------------------------------