import psycopg2 as db
import scipy
import scipy.spatial
import pickle
import time
import json
from psycopg2.extras import Json
from collections import namedtuple
import pprint
import sys
import numpy as np
#(-1509555457L, datetime.date(2013, 10, 27), datetime.date(2013, 10, 30), 
#-33.8562545, 151.20955609, -33.86937624, 151.10843715, 261.089860851442, 
#9450.56626749933, 271804.0, 0.101966759717818, 0.0347697836216514)


pp = pprint.PrettyPrinter(indent=4)

FILE_DIR='./Data/'
fname='teststream%s.csv'
frange=range(34)
#frange=range(1) 

conn = db.connect("host=localhost dbname=aussie_twitter user=postgres password=dbzjun client_encoding='utf-8'")
# conn.set_character_set('utf8')
cur = conn.cursor()
# cur.execute('SET NAMES utf8;')
# cur.execute('SET CHARACTER SET utf8;')
# cur.execute('SET character_set_connection=utf8;')

# cur.execute("""DROP TABLE IF EXISTS new_assuie_tweets""")
# conn.commit()
# cur.execute("""CREATE TABLE new_assuie_tweets ( id serial PRIMARY KEY, data json )""")



CREATE_QUERY="""CREATE TABLE aussie_tweets
(
  latitude double precision,
  longitude double precision,
  has_geotag boolean,
  location_type varchar(16),
  location_name varchar(64),
  created_at timestamp with time zone,
  entities json,
  favorite_count integer,
  in_reply_to json,
  retweet_count integer,
  text text,
  source text,
  timestamp_ms bigint NOT NULL,
  usr json,
  uid bigint NOT NULL,
  id    serial,
  CONSTRAINT pk PRIMARY KEY (id)
)
"""


INSERT_QUERY="""(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""


try:
    #pass
    cur.execute("""DROP TABLE IF EXISTS aussie_tweets""")
    cur.execute(CREATE_QUERY)
    conn.commit()
except db.Error, e:
    print "Error ocurred: %s " % e.args[0]
    print e


line_lim=0
bulk=1000


sub_keys = [ 'in_reply_to_screen_name', 'in_reply_to_status_id', \
    'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str']


count = 0

for i in frange:
    id = '' if i==0 else str(i)
    fn = fname%id

    print '*********************** Working with %s' % fn

    with open(FILE_DIR+fn, 'rU') as f:

        bcount = 0
        buff = []
        for line in f:
            if len(line) < 100:
                continue

            # line = to_utf(line)

            #print 'raw:', line
            
            # line = highpoints.sub(u'', line)

            # print 'converted:', line

            # obj = json.loads(line)

            # print 'dict v', type(obj['text']), obj['text'].encode('utf-8')

            obj = json.loads(line, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

            #print 'obj v', type(obj.text), obj.text.encode('utf-8')


            #pp.pprint(json.loads(line))
            
            # print cur.mogrify(INSERT_QUERY, [obj.coordinates.coordinates[1],obj.coordinates.coordinates[0],obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count,obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id])

            #print obj

            # if obj.place is None and obj.coordinates is None:
            #     continue

            buff.append(obj)
            bcount += 1

            #print obj.geo.coordinates[0]


            if bcount == bulk:
                
                try:
                    #print [ (obj.geo['coordinates'][0],obj.geo['coordinates'][1],obj.created_at,Json(obj.entities),Json(obj.media),obj.favorited_count,Json(obj.in_reply_to),obj.retweet_count,obj.text,obj.#source,obj.timestamp_ms,Json(obj.user),obj.uid) for obj in buff]


                    #print type(obj.text)
                
                    # args_str = ','.join( cur.mogrify(INSERT_QUERY, [obj.geo.coordinates[1],obj.geo.coordinates[0],obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count, obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id]) for obj in buff)

                    #print args_str

                    str_buff = []
                    for obj in buff:
                        has_geotag = obj.coordinates is not None

                        if has_geotag:
                            lat,lon = obj.coordinates.coordinates
                        elif obj.place is not None:
                            lat,lon = np.mean([obj.place.bounding_box.coordinates[0][0][0],obj.place.bounding_box.coordinates[0][2][0]]), np.mean([obj.place.bounding_box.coordinates[0][0][1],obj.place.bounding_box.coordinates[0][1][1]])  
                        else:
                            lat,lon = None,None

                        ptype = obj.place.place_type if obj.place is not None else None
                        pname = obj.place.full_name if obj.place is not None else None


                        str_buff.append(( lat, lon, has_geotag, ptype, pname, obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count, obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id))


                    sql_str = """INSERT INTO aussie_tweets (latitude,longitude,has_geotag,location_type,location_name,created_at,entities,favorite_count,in_reply_to,retweet_count,text,source,timestamp_ms,usr,uid) VALUES """ + ','.join( cur.mogrify(INSERT_QUERY, item) for item in str_buff)

                    # sql_str = """INSERT INTO aussie_tweets (latitude,longitude,created_at,entities,favorite_count,in_reply_to,retweet_count,text,source,timestamp_ms,usr,uid) VALUES """ + ','.join( cur.mogrify(INSERT_QUERY, [obj.geo.coordinates[1],obj.geo.coordinates[0],obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count, obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id]) for obj in buff)
                    #print sql_str
                    cur.execute(sql_str) 
                    conn.commit()
                    #cur.execute("""INSERT INTO new_assuie_tweets (data) VALUES(%s)""",[Json(line)])
                    count += bcount
                    print 'Rows inserted: %d' % count
                except db.Error, e:
                    print "Error ocurred: %s " % e
                    print e.pgerror
                    #print 'query string:', sql_str
                    #sys.exit(1)

                bcount = 0
                buff = []

            
            if line_lim>0 and count > line_lim:
                break

    if line_lim>0 and count > line_lim:
        break

    if bcount > 0:
        try:
            # sql_str = """INSERT INTO aussie_tweets (latitude,longitude,created_at,entities,favorite_count,in_reply_to,retweet_count,text,source,timestamp_ms,usr,uid) VALUES """ + ','.join( cur.mogrify(INSERT_QUERY, [obj.geo.coordinates[1],obj.geo.coordinates[0],obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count, obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id]) for obj in buff)

            str_buff = []
            for obj in buff:
                has_geotag = obj.coordinates is not None

                if has_geotag:
                    lat,lon = obj.coordinates.coordinates
                elif obj.place is not None:
                    lat,lon = np.mean([obj.place.bounding_box.coordinates[0][0][0],obj.place.bounding_box.coordinates[0][2][0]]), np.mean([obj.place.bounding_box.coordinates[0][0][1],obj.place.bounding_box.coordinates[0][1][1]])  
                else:
                    lat,lon = None,None

                ptype = obj.place.place_type if obj.place is not None else None
                pname = obj.place.full_name if obj.place is not None else None


                str_buff.append(( lat, lon, has_geotag, ptype, pname, obj.created_at,Json(obj.entities), obj.favorite_count,Json( { k:obj.__dict__[k] for k in sub_keys } ),obj.retweet_count, obj.text,obj.source,obj.timestamp_ms,Json(obj.user),obj.user.id))


            sql_str = """INSERT INTO aussie_tweets (latitude,longitude,has_geotag,location_type,location_name,created_at,entities,favorite_count,in_reply_to,retweet_count,text,source,timestamp_ms,usr,uid) VALUES """ + ','.join( cur.mogrify(INSERT_QUERY, item) for item in str_buff)

            #print args_str
            cur.execute(sql_str) 
            conn.commit()
            #cur.execute("""INSERT INTO new_assuie_tweets (data) VALUES(%s)""",[Json(line)])
            count += bcount
        except db.Error, e:
            print "Error ocurred: %s " % e.args[0]
            print e.pgerror
            #print 'query string:', sql_str
        buff = []
        bcount = 0

print 'total rows inserted:', count



