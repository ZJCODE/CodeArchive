# -*- coding: utf-8 -*-
import time
from datetime import datetime
from collections import Counter

# open train data
train = open('../data/train.csv','r')

# set file paths 

loc_start_end_pair_path = '../data/loc_start_end_pair'
user_places_path = '../data/user_places_all'

user_places_pair_path = '../data/graph_user_places_pair'
user_start_places_pair_path = '../data/graph_user_start_places_pair'
user_end_places_pair_path = '../data/graph_user_end_places_pair'

user_places_hour_1_8_workday_path = '../data/user_places_hour_1_8_workday'
user_places_hour_9_16_workday_path = '../data/user_places_hour_9_16_workday'
user_places_hour_17_24_workday_path = '../data/user_places_hour_17_24_workday'

user_places_hour_1_8_weekday_path = '../data/user_places_hour_1_8_weekday'
user_places_hour_9_16_weekday_path = '../data/user_places_hour_9_16_weekday'
user_places_hour_17_24_weekday_path = '../data/user_places_hour_17_24_weekday'



# open files 
loc_start_end_pair = open(loc_start_end_pair_path,'w')
user_places = open(user_places_path,'w')

user_places_pair = open(user_places_pair_path,'w')
user_start_places_pair = open(user_start_places_pair_path,'w')
user_end_places_pair = open(user_end_places_pair_path,'w')

user_places_hour_1_8_workday = open(user_places_hour_1_8_workday_path,'w')
user_places_hour_9_16_workday = open(user_places_hour_9_16_workday_path,'w')
user_places_hour_17_24_workday = open(user_places_hour_17_24_workday_path,'w')

user_places_hour_1_8_weekday = open(user_places_hour_1_8_weekday_path,'w')
user_places_hour_9_16_weekday = open(user_places_hour_9_16_weekday_path,'w')
user_places_hour_17_24_weekday = open(user_places_hour_17_24_weekday_path,'w')

# process data

# hour distribution :
'''
Counter({0: 13306,1: 6216,2: 3714,3: 2889,4: 4397,5: 29082,6: 121711,7: 316142,8: 285078,9: 148047,
         10: 126796,11: 172365,12: 199831,13: 163768,14: 136380,15: 155011,16: 179272,17: 273240,
         18: 290294,19: 209194,20: 151748,21: 127221,22: 69904,23: 28489})
'''


head = train.readline().strip().split(',')
start_end_list = []

user_places_list = []
user_start_places_list = []
user_end_places_list = []

user = {}
user_hour_1_8_workday = {}
user_hour_9_16_workday = {}
user_hour_17_24_workday = {}
user_hour_1_8_weekday = {}
user_hour_9_16_weekday = {}
user_hour_17_24_weekday = {}

cnt = 0
for line in train.readlines():
    cnt += 1
    if cnt % 50000 == 0:
        print 'process line %d'%(cnt)
    orderid,userid,bikeid,biketype,start_time,loc_start,loc_end = line.strip().split(',')
    dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    # start_end pair
    start_end_list.append(loc_start+'\t'+loc_end)

    # user_place_pair 
    user_places_list.append(userid+'\t'+loc_start)
    user_places_list.append(userid+'\t'+loc_end)
    user_places_list.append(userid+'\t'+loc_start + '-' + loc_end)
    user_start_places_list.append(userid+'\t'+loc_start)
    user_end_places_list.append(userid+'\t'+loc_end)

    # user places
    if userid not in user:
        user[userid] = []
        user[userid].append(loc_start+'-'+loc_end)
    else:
        user[userid].append(loc_start+'-'+loc_end)
    # week 0-6 | 0:monday

    # user 1-8 workday places files
    if dt.hour >=1 and dt.hour <=8 and dt.weekday() < 5:
	if userid not in user_hour_1_8_workday :
	    user_hour_1_8_workday[userid] = []
	    user_hour_1_8_workday[userid].append(loc_start+'-'+loc_end)
	else:
	    user_hour_1_8_workday[userid].append(loc_start+'-'+loc_end)
    # user 9-16 workday places files
    elif dt.hour >= 9 and dt.hour <= 16 and dt.weekday() < 5:
        if userid not in user_hour_9_16_workday :
            user_hour_9_16_workday[userid] = []
            user_hour_9_16_workday[userid].append(loc_start+'-'+loc_end)
        else:
            user_hour_9_16_workday[userid].append(loc_start+'-'+loc_end)
   # user 17-24 workday places files
    elif (dt.hour >= 17 or dt.hour == 0) and dt.weekday() < 5:
        if userid not in user_hour_17_24_workday:
            user_hour_17_24_workday[userid] = []
            user_hour_17_24_workday[userid].append(loc_start+'-'+loc_end)
        else:
            user_hour_17_24_workday[userid].append(loc_start+'-'+loc_end)
    # user 1-8 weekday places files
    elif dt.hour >=1 and dt.hour <=8 and dt.weekday() >= 5:
        if userid not in user_hour_1_8_weekday :
            user_hour_1_8_weekday[userid] = []
            user_hour_1_8_weekday[userid].append(loc_start+'-'+loc_end)
        else:
            user_hour_1_8_weekday[userid].append(loc_start+'-'+loc_end)
    # user 9-16 weekday places files
    elif dt.hour >= 9 and dt.hour <= 16 and dt.weekday() >= 5:
        if userid not in user_hour_9_16_weekday :
            user_hour_9_16_weekday[userid] = []
            user_hour_9_16_weekday[userid].append(loc_start+'-'+loc_end)
        else:
            user_hour_9_16_weekday[userid].append(loc_start+'-'+loc_end)
    # user 17-24 weekday places files
    elif (dt.hour >= 17 or dt.hour == 0) and dt.weekday() >= 5:
        if userid not in user_hour_17_24_weekday:
            user_hour_17_24_weekday[userid] = []
            user_hour_17_24_weekday[userid].append(loc_start+'-'+loc_end)
        else:
            user_hour_17_24_weekday[userid].append(loc_start+'-'+loc_end)

# write loc_start_end_pair
print 'write loc_start_end_pair'
pair_count = Counter(start_end_list).most_common()
for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    loc_start_end_pair.write(line)
loc_start_end_pair.close()
del start_end_list
del pair_count


#user_place_pair 
print 'write user_places'
pair_count = Counter(user_places_list).most_common()
for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    user_places_pair.write(line)
user_places_pair.close()
del user_places_list
del pair_count


print 'write user_start_places'
pair_count = Counter(user_start_places_list).most_common()
for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    user_start_places_pair.write(line)
user_start_places_pair.close()
del user_start_places_list
del pair_count


print 'write user_end_places'
pair_count = Counter(user_end_places_list).most_common()
for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    user_end_places_pair.write(line)
user_end_places_pair.close()
del user_end_places_list
del pair_count



# write user places
print 'write user places'
for k,v in user.items():
    user_places.write(k+'|'+'\t'.join(v)+'\n')
user_places.close()
del user

# write user 1-8 workday places
print 'write user 1-8 workday places'
for k,v in user_hour_1_8_workday.items():
    user_places_hour_1_8_workday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_1_8_workday.close()
del user_hour_1_8_workday

# write user 9-16 workday places files
print 'write user 9-16 workday places files'
for k,v in user_hour_9_16_workday.items():
    user_places_hour_9_16_workday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_9_16_workday.close()
del user_hour_9_16_workday

# write user 17-24 workday places files
print 'write user 17-24 workday places files'
for k,v in user_hour_17_24_workday.items():
    user_places_hour_17_24_workday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_17_24_workday.close()
del user_hour_17_24_workday

# write user 1-8 weekday places files
print 'write user 1-8 weekday places files'
for k,v in user_hour_1_8_weekday.items():
    user_places_hour_1_8_weekday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_1_8_weekday.close()
del user_hour_1_8_weekday

# write user 9-16 weekday places files
print 'write user 9-16 weekday places files'
for k,v in user_hour_9_16_weekday.items():
    user_places_hour_9_16_weekday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_9_16_weekday.close()
del user_hour_9_16_weekday

# write user 17-24 weekday places files
print 'write user 17-24 weekday places files'
for k,v in user_hour_17_24_weekday.items():
    user_places_hour_17_24_weekday.write(k+'|'+'\t'.join(v)+'\n')
user_places_hour_17_24_weekday.close()
del user_hour_17_24_weekday

print 'done!'

