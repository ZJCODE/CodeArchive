from urllib2 import *
import simplejson
import webbrowser
from gensim import corpora, models, similarities
from bs4 import BeautifulSoup


def Search(QueryWords,re_rank):
    import time
    t1 = time.time()
    #### Process QueryWordsList (fliter stopwords)####
    stopwords = []
    with open('stopword_eng.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        stopwords.append(line.lower().strip())
    qw = QueryWords.lower().strip().split(' ')
    qw = [w for w in qw if w not in stopwords]
    QueryWords = ','.join(qw)
    ####  Get solr result ####
    load_pages = 300
    url = 'http://localhost:8983/solr/gettingstarted/select?indent=on&q=' + str(QueryWords)+ '&rows='+ str(load_pages)+ '&start=0&wt=json'
    conn = urlopen(url)

    rsp = simplejson.load(conn)
    t2 = time.time() 
    length = len(rsp['response']['docs'])
    #print 'length: ' + str(length)+'\n'
    if length > 100:
        print '\nMore than 100 Result are found , Cost : ' + str(round((t2-t1),5)) + ' seconds \n' 
    else:
        print '\nThere are only ' + str(length) +' Results , Cost : ' + str(round((t2-t1),5))  + ' seconds \n' 
        if length == 0 :
            return 0
    
    ####    Re-Rank Process   #####

    '''
    using corpus to vector to calculate similarity [gensim]
    '''
    def doc_sim(texts,query):
        '''
        texts like this ['sdsa asd asd','asdas asdasd sdas']
        query like this 'dsd,dsf'
        '''
        original_texts = texts[:]
        texts = [t.lower().replace('_',' ').split() for t in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        vec_bow = dictionary.doc2bow(query.lower().split(','))
        index = similarities.MatrixSimilarity(corpus)
        sims = index[vec_bow]
        sims_num = list(enumerate(sims))
        sr = sorted(sims_num,key=lambda x :x[1],reverse=True)
        order = [x[0] for x in sr]
        if re_rank == 'yes' or re_rank == 'y':            
            output = [original_texts[i] for i in order]
        else:
            output = original_texts
        return output

    page_titles = []
    Index = []
    for i in range(load_pages):
        try:
            title = str(rsp['response']['docs'][i]['title'][0][:-35])
            if title not in page_titles:                
                page_titles.append(title)
                Index.append(i)
        except:
            pass
    
    dict_title_index = dict(zip(page_titles,Index))
    titles_re_rank = doc_sim(page_titles,QueryWords)
    Index_re_rank = [dict_title_index[t] for t in titles_re_rank]

    
    def GetBrief(url,title):
        html = urlopen(url).read()
        bsObj = BeautifulSoup(html,'lxml')
        t = bsObj.text
        title = title[:20]
        show = [c for c in re.findall(title.lower()+'.*',t.lower()) if len(c)>100][0]#[:200]
        if len(show) == 0 :
            show = [c for c in re.findall(title.split()[0].lower()+'.*',t.lower()) if len(c)>100][0]#[:200]
        return show
    

    
    #####  control the number of pages to see  #####
    view_num = input('How many pages you want to see ? ')
    
    ####  show the title and abstract   ####
    for page_num,i in enumerate(Index_re_rank[:view_num]):
        try:
            print '--------------------------------------------------------------'
            print '\n['+str(page_num+1)+']  \n\n' +'title:   ' + str(rsp['response']['docs'][i]['title'][0][:-35]) + '\n'
            # print 'Brief View:  \n\n' + str(rsp['response']['docs'][i]['keywords'][0]) + '\n\n'   
            print 'Brief View:  \n'
            url = 'file://' + str(rsp['response']['docs'][i]['id'])
            title = str(rsp['response']['docs'][i]['title'][0])[:-35]
            print GetBrief(url,title)
            print '\n'
        except:
            pass
    #### open html page #####
    while True:
        num = input('Input The Num of Page you want to see or input \'0\' to stop :   ')         
        if num != 0:
            loc = 'file://' + str(rsp['response']['docs'][Index_re_rank[num-1]]['id'])
            webbrowser.open(loc)
        else:
            break



logo = '''
               #    ######    ######        #         ######    #######    #      #
                   #         #            # #        #    #    #          #      #
             #    #         #           #   #       #    #    #          #      #
            #    ######    ######     #     #      ######    #          ########
           #         #    #          # # # #      # #       #          #      #
          #         #    #         #       #     #   #     #          #      #
         #    ######    ######   #         #    #     #   #######    #      #
'''



if __name__ == '__main__':
    import os
    while True:
        os.system('clear')
        print logo
        #re_rank = input('Use re_rank or not ?  [yes/no] : ')
        Q = raw_input('\nInput The Query Word :  ' )
        if Q == 'quit()':
            os._exit(1) 
        Search(Q,re_rank = 'no')


        