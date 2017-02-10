import sys

Word = sys.argv[1]

Num = sys.argv[2]

import urllib
import bs4 #this is beautiful soup

url = 'http://www.wordnik.com/words/' + str(Word)
source = urllib.urlopen(url).read()
soup = bs4.BeautifulSoup(source,"lxml")

Def = soup.find_all('div',{'class':'word-module module-definitions'})[0]

definition = Def.find_all('li')

A=[a.getText() for a in definition][:int(Num)]

print('\n\nThe Meaning of '+str(Word)+' :\n')
for i,a in enumerate(A):
    print(str(1+i)+ ':  ' + a)

print('\n\n')

