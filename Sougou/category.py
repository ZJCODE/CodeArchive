# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:42:13 2016

@author: ZJun
"""

import glob
import jieba
import pandas as pd

def SplitText(Sentence):
    return list(jieba.cut(Sentence, cut_all=False))
    
floders = glob.glob('./SougouCategory/*')

categorys_w = []

for floder in floders:
    print '==========process category' + floder[17:]+'=========='
    path = floder + '/*.txt'
    files = glob.glob(path)
    
    category_words = []
    for f in files:
        words =[]
        with open(f,'r') as txt:
            print '==========Process files' + f +'=========='
            lines = txt.readlines()
            for line in lines:
                words += SplitText(line.strip())
        category_words += words
    
    categorys_w.append(category_words)

category_list = [c[17:] for c in floders]

dict_category_word = dict(zip(category_list,categorys_w))
df_category_word = pd.DataFrame({'category':category_list,'words':categorys_w})
df_category_word.to_csv('df_category_word.csv',index=False,encoding='utf8')



        
        
