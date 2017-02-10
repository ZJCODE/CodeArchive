# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:06:59 2016

@author: Admin
"""

import numpy as np
import math
Xdata = DC.Degree
Ydata = DC.Cluster


Xdata_log = np.log(Xdata[1:])
Ydata_log = np.log(Ydata[1:])



 
#Xdata_log = np.log(X_Log_Bin)
#Ydata_log = np.log(Y_Log_Bin)
 
k = float(np.dot(Xdata_log,Ydata_log)) / np.dot(Xdata_log,Xdata_log)
b = np.mean(Ydata_log) - k * np.mean(Xdata_log)

	
x = Xdata
def fun(x,k,b):
    return math.exp(b) * math.pow(x,k)
y0 =[fun(i,k,b) for i in x]
y =[fun(i,2,b) for i in x]
yy =[fun(i,1,b) for i in x]
plt.cla()
plt.loglog(Xdata,Ydata,'ko',alpha = 0.3,markersize=3)
#plt.loglog(x,y0,'k--')
plt.loglog(x[3:400],y[3:400],'k--')
plt.loglog(x[900:-1],yy[900:-1],'k--')
#plt.text(10000,4000000,'k=1.5')
plt.text(10,8000,'slope=2')
plt.text(10000,2000,'slope=1')
plt.xlabel('K',fontsize = 13)
plt.ylabel(r'$\pi\left(k\right)$',fontsize = 15)
plt.title('Measuring Preferential Attachment' , fontsize  =13)


def G_N(List):
    n_max = max(List)
    n_min = min(List)
    n_mean =sum(List)/(1.0*len(List))
    L = [(i-n_mean)*1.0 / (n_max-n_min) for i in List]
    return L

plt.plot(range(len(Graph.Num_Node)),G_N(Graph.Num_Node),'o')
plt.plot(range(len(Graph.Num_Node)),G_N(Graph.k_Max),'>')
plt.plot(range(len(Graph.Num_Node)),G_N(Graph.k_Mean),'s')









fig, axes = plt.subplots(3, 2, figsize=(12, 9))   # 构建绘图框架

fig.subplots_adjust(hspace=0.5, wspace=0.3)


plt.subplot(321)
plt.plot(range(len(C)),C.N,'-o',markersize=2)
plt.title('Evolution of Node Number')
plt.xlabel('Month')
plt.ylabel('Node Number')

plt.subplot(322)
plt.plot(range(len(C)),C.Mean,'-o',markersize=2)
plt.title('Evolution of Average Degree')
plt.xlabel('Month')
plt.ylabel('Average Degree')



plt.subplot(323)
plt.plot(range(len(C)),C.G,'-o',markersize=2)
plt.ylim([1.4,1.7])
plt.title('Evolution of Power Law Coef')
plt.xlabel('Month')
plt.ylabel('Power Law Coef')


plt.subplot(324)
plt.plot(range(len(C)),C.A,'-o',markersize=2)
plt.ylim([-0.350,0])
plt.title('Evolution of Assortativity Coef')
plt.xlabel('Month')
plt.ylabel('Assortativity Coef')

plt.subplot(325)
plt.plot(range(len(C)),C.AC,'-o',markersize=2)
plt.ylim([0,0.6])
plt.title('Evolution of Average Clustering Coef')
plt.xlabel('Month')
plt.ylabel('Average Clustering Coef')

plt.subplot(326)
plt.plot(range(len(C)),C.AP,'-o',markersize=2)
plt.ylim([2,4])
plt.title('Evolution of Average Path Length')
plt.xlabel('Month')
plt.ylabel('Average Path Length')

#---------------------------------------------------------------


T = pd.read_csv('Time_Interval.txt')
T = T[T.Time_Interval>0]
T.index = range(len(T))
CT = Counter(T.Time_Interval)
SCT = sorted(list(CT.items()),key = lambda x:x[0] )

L = []
Sum = sum(CT.values())
for alpha in range(10):
    S = 0
    for i in range(len(SCT)):
        S = S + SCT[i][1]
        if S > alpha /10.0 * Sum:
            break
    L.append(i)

# 22419 0.9
# 4552 0.8
# 1582 0.7
# 830 0.6
# 362 0.5

Time = [i[0] for i in SCT]
Num = [i[1] for i in SCT]

plt.plot(Time,Num,'--o',color='k',alpha=0.8,markersize = 3)
plt.xlim([0,1440])
plt.title('Time Interval Distribution',fontsize = 14)
plt.xlabel('Minutes',fontsize = 13)
plt.ylabel('The Amount of Reply',fontsize = 13)

plt.loglog(Time,Num,'--o',color='k',alpha=0.8)
    
X = Historgam_List(I.Life_Span,60*24)
X_Index = range(1,len(X)+1)
Pair = zip(X_index,X)
NoZeroPair = [i for i in Pair if i[1] != 0]
Xdata,Ydata = zip(*NoZeroPair)
plt.loglog(Xdata[0],Ydata[0],'ro',alpha=0.5,markersize = 15)
plt.loglog(Xdata[1:],Ydata[1:],'go',alpha=0.3)
plt.title('Online Life Span Distribution',fontsize=13)
plt.xlabel('Day',fontsize=13)
plt.ylabel('Amount',fontsize=13)
plt.xlim([-100,10e5])
plt.ylim([-10e2,10e5])




N=20000


Hub = ['whatUwant','42','sepheric']

Active_List_Hub = [list(I[I.ID == Id].Active_Time_List)[0] for Id in Hub]

Interval = [np.diff(np.array(L)) for L in Active_List_Hub]

fig, axes = plt.subplots(1, 3, figsize=(10, 3))   # 构建绘图框架

fig.subplots_adjust(hspace=0.5, wspace=0.4)

plt.subplot(131)
plt.plot(range(len(Interval[0])),Interval[0])
plt.ylim([0,N])
plt.xlim([0,15000])
plt.xticks(range(min(range(len(Interval[0]))),max(range(len(Interval[0])))+1,4000))
plt.ylabel('Time Interval')
plt.xlabel('Sequence of Activities')
plt.title(Hub[0])

plt.subplot(132)
plt.plot(range(len(Interval[1])),Interval[1])
plt.ylim([0,N])
plt.xlim([0,27000])
plt.xticks(range(min(range(len(Interval[1]))),max(range(len(Interval[1])))+1,6000))
plt.xlabel('Sequence of Activities')
plt.ylabel('Time Interval')
plt.title(Hub[1])


plt.subplot(133)
plt.plot(range(len(Interval[2])),Interval[2])
plt.ylim([0,N])
plt.xticks(range(min(range(len(Interval[2]))),max(range(len(Interval[2])))+1,10000))
plt.xlabel('Sequence of Activities')
plt.ylabel('Time Interval')
plt.title(Hub[2])


People = ['ocean159','hengry129','ruanyutai']

Active_List_Hub = [list(I[I.ID == Id].Active_Time_List)[0] for Id in People]

Interval = [np.diff(np.array(L)) for L in Active_List_Hub]

fig, axes = plt.subplots(1, 3, figsize=(10, 3))   # 构建绘图框架

fig.subplots_adjust(hspace=0.5, wspace=0.4)


plt.subplot(131)
plt.plot(range(len(Interval[0])),Interval[0])
plt.ylim([0,N])
plt.xticks(range(min(range(len(Interval[0]))),max(range(len(Interval[0])))+1,100))
plt.xlabel('Sequence of Activities')
plt.ylabel('Time Interval')
plt.title(People[0])

plt.subplot(132)
plt.plot(range(len(Interval[1])),Interval[1])
plt.ylim([0,N])
plt.xticks(range(min(range(len(Interval[1]))),max(range(len(Interval[1])))+1,20))
plt.xlabel('Sequence of Activities')
plt.ylabel('Time Interval')
plt.title(People[1])

plt.subplot(133)
plt.plot(range(len(Interval[2])),Interval[2])
plt.ylim([0,N])
plt.xticks(range(min(range(len(Interval[2]))),max(range(len(Interval[2])))+1,30))
plt.xlabel('Sequence of Activities')
plt.ylabel('Time Interval')
plt.title(People[2])
