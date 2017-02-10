import pandas as pd 
import numpy as np
import MySQLdb

def mysql_into():
	mysql_cn= MySQLdb.connect(host='localhost', port=3306,user='root', passwd='fei199418forever', db='taobao')
 	data_12_18 = pd.read_sql("select * from u where left(time,10) = '2014-12-18' order by time;", con=mysql_cn) 
	data_p = pd.read_sql("select item_id from p ;", con=mysql_cn)
	mysql_cn.close()
 	df = pd.merge(data_12_18, data_p, on='item_id')
	return df

def dell_df(df):
	df['num'] = [1]*len(df)
	user_item_id = [str(u)+'-'+str(i) for u,i in zip(df.user_id, df.item_id)]
	df['user_item'] =  user_item_id
	return df

def pivot_df(df):
	attr = df.pivot_table(values='num', index='user_item',columns='behavior_type', aggfunc=np.sum, fill_value=0)
	return attr


df = mysql_into()
df = dell_df(df)
df = pivot_df(df)


def chose_df(df):
	df = df[(df['1'] >= 4) & (df['4'] == 0) & (df['3'] > 0)]
	df['user_item'] = df.index
	return df

def split_df_user_item(df):
	user_id = [x.split('-')[0]  for x in df['user_item']]
	item_id = [x.split('-')[1]  for x in df['user_item']]
	df['user_id'] = user_id
	df['item_id'] = item_id
	return df[['user_id','item_id']]

def save(df, path):
	df.to_csv(path, sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

path = '/home/cookly/taobao_9.22/2day/tianchi_mobile_recommendation_predict.csv'

df = chose_df(df)
df = split_df_user_item(df)
save(df, path)







