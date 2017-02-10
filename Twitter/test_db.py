import psycopg2 as db
import scipy
import scipy.spatial
import pickle
import numpy as np

conn = db.connect("host=localhost dbname=aussie_twitter user=postgres password=dbzjun client_encoding='utf-8'")
# conn.set_character_set('utf8')
cur = conn.cursor()


SELECT_QUERY="""SELECT latitude, longitude, text , timestamp_ms FROM aussie_tweets where has_geotag=true or location_type='neighborhood'"""


try:
    cur.execute(SELECT_QUERY)
    rows = cur.fetchall()
    X = np.array(rows)
except db.Error, e:
    print "Error ocurred: %s " % e.args[0]
    print e
