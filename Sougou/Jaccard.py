# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:32:43 2016

@author: ZJun
"""



def Import_Obj(File):    
    import pickle
    File_Name = File+'.pkl'
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)
    
    
def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()



def Jaccard(corpus1,corpus2):
    intersection = set(corpus1) & set(corpus2)
    union = set(corpus1) | set(corpus2)
    return len(intersection)*1.0 / len(union)
  
'''  
t1 = time.time()
j = Jaccard(TrainData.QueryWordsList[3],TestData.QueryWordsList[5])
t2 = time.time()
print t2-t1
'''

Tag = []
for corpus1 in TestData.QueryWordsList:
    j = -1
    tagpre = '0'
    for corpus2,tag in zip(TrainData.QueryWordsList,TrainData.ModifyTag):
        jaccard = Jaccard(corpus1,corpus2)
        if jaccard > j:
            tagpre = tag
            j = jaccard