# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:20:14 2016

@author: ZJun
"""


import warc
f = warc.open("00.warc.gz")
f.read_record()
i = 0
for record in f:
    i = i+1
    print record['WARC-Target-URI'], record['Content-Length']
    if i >10:
        break
f.close()


f = warc.open("00.warc.gz")
a = f.read_record()
b = f.read_record()

'''
a.header.items()
Out[52]: 
[('warc-type', 'warcinfo'),
 ('content-length', '219'),
 ('version', '0.18'),
 ('warc-date', '2009-03-75T00:59:24-0400'),
 ('content-type', 'application/warc-fields'),
 ('warc-record-id', '<urn:uuid:b38cd8ab-5ba6-445c-9c9c-0a5cbc3b6a41>')]
 
 
b.header.items()
Out[53]: 
[('content-length', '58485'),
 ('warc-warcinfo-id', 'b38cd8ab-5ba6-445c-9c9c-0a5cbc3b6a41'),
 ('warc-date', '2009-02-51T21:29:14-0500'),
 ('warc-identified-payload-type', 'text/html; charset=utf-8'),
 ('version', '0.18'),
 ('warc-target-uri', 'http://en.wikipedia.org/wiki/\x00pirana_Ngata'),
 ('content-type', 'application/http;msgtype=response'),
 ('warc-record-id', '<urn:uuid:e1457122-2f7e-43b0-99b0-86fa74c4d00c>'),
 ('warc-trec-id', 'clueweb09-enwp00-00-00000'),
 ('warc-type', 'response')]
 
 
b.header.get('warc-target-uri')
Out[138]: 'http://en.wikipedia.org/wiki/\x00pirana_Ngata'
'''

obj = f.fileobj


