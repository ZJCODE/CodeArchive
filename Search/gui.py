from Tkinter import *

from urllib2 import *
import simplejson
import webbrowser

def Search(QueryWords):
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
    url = 'http://localhost:8983/solr/gettingstarted/select?indent=on&q=' + str(QueryWords)+ '&rows=200&start=0&wt=json'
    conn = urlopen(url)
    rsp = simplejson.load(conn)
    
    length = len(rsp['response']['docs'])
        
    if length > 10:
    	pages_num = 5
    else:
    	pages_num = length

    page_titles = set()
    Index = []
    i=0
    while len(page_titles) < pages_num:
        try:
            title = str(rsp['response']['docs'][i]['title'][0])
            if title not in page_titles:                
                page_titles.add(title)
                Index.append(i)
        except:
            pass
        i = i + 1

    
    ####  show the title and abstract   ####
    output = ''
    for page_num,i in enumerate(Index):
        try:
            #output = output + '--------------------------------------------------------------\n'
            output = output + '\n['+str(page_num+1)+']  \n\n' +'title:   ' + str(rsp['response']['docs'][i]['title'][0]) + '\n'
            #output = output + 'Brief View:  \n\n' + str(rsp['response']['docs'][i]['keywords'][0]) + '\n\n'     
        except:
        	pass
        #loc = 'file://' + str(rsp['response']['docs'][Index[num-1]]['id'])
    return output

def submit():
   print(u.get())
   lab2 = Label(frame, text=Search(u.get()))
   lab2.grid(row=0, column=10, padx=10, pady=10, sticky=W)

root = Tk()
root.title("iSearch")
frame = Frame(root)
frame.pack(padx=0, pady=0, ipadx=0)
lab1 = Label(frame, text="get:")
lab1.grid(row=0, column=0, padx=5, pady=5, sticky=W)
u = StringVar()
ent1 = Entry(frame, textvariable=u)
ent1.grid(row=0, column=1, sticky='ew', columnspan=2)



button = Button(frame, text="Search", command=submit, default='active')
button.grid(row=2, column=1)




root.update_idletasks()
x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2 *10
y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2 *10
root.geometry("+%d+%d" % (x, y))
root.mainloop()