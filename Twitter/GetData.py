import psycopg2 as db
import numpy as np
import pandas as pd

def GetData(with_geo_tag = True):
    
    conn = db.connect("host=localhost dbname=aussie_twitter user=postgres password=dbzjun client_encoding='utf-8'")
    # conn.set_character_set('utf8')
    cur = conn.cursor()
    
          
    if with_geo_tag == True :
        SELECT_QUERY="""
        SELECT uid,latitude, longitude, text , created_at 
        FROM aussie_tweets 
        where has_geotag=true 
        ORDER BY created_at
        """
#and created_at BETWEEN '2015-01-12 00:00:00' AND '2015-02-12 00:00:00'

    else:        
        SELECT_QUERY="""
        SELECT uid,latitude, longitude,location_name, text , created_at 
        FROM aussie_tweets 
        ORDER BY created_at
        """ 
    
    
    try:
        cur.execute(SELECT_QUERY)
        rows = cur.fetchall()
        X = np.array(rows)
    except db.Error, e:
        print "Error ocurred: %s " % e.args[0]
        print e
      
    Data = pd.DataFrame(X,columns=['uid','latitude', 'longitude','text' , 'created_at'])
    return Data
    
    
    
def getIdNameDict():
    conn = db.connect("host=localhost dbname=aussie_twitter user=postgres password=dbzjun client_encoding='utf-8'")
    # conn.set_character_set('utf8')
    cur = conn.cursor()
    
    SELECT_QUERY="""
    SELECT usr,uid,created_at 
    FROM aussie_tweets 
    ORDER BY created_at
    """

    try:
        cur.execute(SELECT_QUERY)
        rows = cur.fetchall()
        X = np.array(rows)
    except db.Error, e:
        print "Error ocurred: %s " % e.args[0]
        print e    
    
    Data = pd.DataFrame(X,columns=['usr','uid', 'created_at'])
    user_name = [str.lower(str(u[-9])) for u in Data.usr]
    Data['user_name'] = user_name
    IdName = zip(Data.uid.values,Data.user_name.values)    
    IdNameList = list(set(IdName))
    IdNameDict = dict(IdNameList)
    return IdNameDict


def AddPos(Data):
    PosData = pd.read_csv('./Data/PosData.csv')
    Data['Location'] = PosData.Location
    Data['District'] = PosData.District
    Data = Data.dropna(how='any')
    location_name = [loc+','+dis for loc,dis in zip(Data.Location,Data.District)]
    Data['location_name'] = location_name
    return Data
    

from twitter_function import AddDate,AddUserName        
        
def LoadData():
    Data = GetData()
    Data = AddPos(Data)
    AddDate(Data)
    AddUserName(Data)
    return Data
    