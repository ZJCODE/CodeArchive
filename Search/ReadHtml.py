import warc
import glob
import os.path
import urllib
import string
files = glob.glob('./Data/enwp00/*.*')
print 'Process those files : \n'
print files
#a = input('Enter 1 to continue : ')
for file_ in files:    
    f = warc.open(file_)
    i = 0
    while True:
        i = i+1
        print i
        try:
            look = f.read_record()
            file_path = './'+file_.split('/')[2] +'_'+ file_.split('/')[-1].split('.')[0]+'/'
            html_name = urllib.unquote(look.url).split('/')[-1].strip() + '.html'
            if str.isdigit(html_name[0]) or str.isalpha(html_name[0]) :
                filename = file_path+ html_name
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                print filename
                look_html = look.payload
                with open(filename,'w') as f_html:
                    f_html.write(look_html)
            else:
                pass            

        except:
        	print 'error'
        	if i > 30000:
        		break
       		else:
       			pass