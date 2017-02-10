f = open('ALL_USER_YONGDIAN_DATA.csv','r')
head = f.readline().replace('"','').strip().split(',')
users = []
import pandas as pd
import numpy as np

def ProcessLine(line):
	
	CONS_NO,DATA_DATE,KWH_READING,KWH_READING1,KWH = line.replace('"','').strip().split(',')
	
	DATA_DATE = pd.to_datetime(DATA_DATE)
	
	try:
		KWH_READING = int(KWH_READING)
	except:
		KWH_READING = np.nan

	try:
		KWH_READING1 = int(KWH_READING1)
	except:
		KWH_READING1 = np.nan

	try:
		KWH = int(KWH)
	except:
		KWH = np.nan

	return CONS_NO,DATA_DATE,KWH_READING,KWH_READING1,KWH

while True :
	l = f.readline()
	if len(l) < 1:
		break 
	else:
	 	dosomething


