import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


#--load data-

print 'load data'

test = pd.read_csv('../data/test_new.csv')
train = pd.read_csv('../data/train_new.csv')
'''
user_app_actions = pd.read_csv('../data/user_app_actions_new.csv')
user_installedapps = pd.read_csv('../data/user_installedapps_new.csv')
ad = pd.read_csv('../data/ad.csv')
'''

advertiserID_day_conversion = train.pivot_table('label','advertiserID','day',aggfunc='sum').fillna(0)

advertiserID_bad = set(advertiserID_day_conversion.index[(advertiserID_day_conversion[30] == 0) | (advertiserID_day_conversion[30]/advertiserID_day_conversion[29] <0.5)])

def advertiserID_bad_or_not(advertiserID):
    if advertiserID in advertiserID_bad:
        return 1
    else:
        return 0

train['advertiserID_bad'] = train.advertiserID.apply(advertiserID_bad_or_not)

train = train[~((train.advertiserID_bad==1)&(train.day==30))]


del train['advertiserID_bad']

test.instanceID.to_csv('test_InstanceID.csv')

del test['instanceID']
del train['conversionTime']


data = pd.concat([train,test])
#data = data[data.day>19].reset_index(drop=1)



#---------------------------value&category2value_feature----------------------------------------------


category2value_feature_name = [ 'clickTime',
 'creativeID',
 'userID',
 'positionID',
 'connectionType',
 'telecomsOperator',
 'adID',
 'camgaignID',
 'advertiserID',
 'appID',
 'appPlatform',
 'appCategory',
 'cate',
 'sitesetID',
 'positionType',
 'age',
 'gender',
 'education',
 'marriageStatus',
 'haveBaby',
 'hometown',
 'residence',
 'residence_short',
 'hometown_short',
 'day',
 'click_hour',
 'most_love_have_cate',
 'second_love_have_cate',
 'age_bin',
 'gender_age',
 'positionID_sitesetID',
 'positionType_sitesetID_appID',
 'sitesetID_positionType',
 'positionID_appID',
 'positionID_connectionType',
 'appID_age_bin',
 'appID_gender']


alpha = input('alpha: ')
beta = input('beta: ')


def show_conversion_feature_old(data,day,feature_name,look_back):
    before_data = data[(data.day<day)&(data.day>=(day-look_back))]
    show = before_data.pivot_table('label',feature_name,aggfunc='count')
    conversion = before_data.pivot_table('label',feature_name,aggfunc='sum')
    show_df = pd.DataFrame({feature_name:show.index,feature_name+'_show_cnt':show.values})
    conversion_df = pd.DataFrame({feature_name:conversion.index,feature_name + '_conversion_cnt':conversion.values})
    s_c_feature = show_df.merge(conversion_df,left_on=feature_name,right_on =feature_name,how='left')
    s_c_feature[feature_name+'_conversion_rate'] = s_c_feature[feature_name + '_conversion_cnt']/s_c_feature[feature_name+'_show_cnt']
    return s_c_feature


def show_conversion_feature(data,day,feature_name,look_back):
    before_data = data[(data.day<day)&(data.day!=30)]
    show = before_data.pivot_table('label',feature_name,aggfunc='count')
    conversion = before_data.pivot_table('label',feature_name,aggfunc='sum')
    show_df = pd.DataFrame({feature_name:show.index,feature_name+'_show_cnt':show.values})
    conversion_df = pd.DataFrame({feature_name:conversion.index,feature_name + '_conversion_cnt':conversion.values})
    s_c_feature = show_df.merge(conversion_df,left_on=feature_name,right_on =feature_name,how='left')
    s_c_feature[feature_name+'_conversion_rate'] = (s_c_feature[feature_name + '_conversion_cnt']+alpha)/(s_c_feature[feature_name+'_show_cnt']+alpha+beta)
    #s_c_feature[feature_name+'_conversion_rate'] = s_c_feature[feature_name + '_conversion_cnt']/s_c_feature[feature_name+'_show_cnt']
    return s_c_feature


def old_user(data,day):
    before_data = data[(data.day<day)]
    user_list = before_data.userID.unique()
    user_df = pd.DataFrame({'userID':user_list,'old_user':[1]*len(user_list)})
    return user_df

def old_user_conv(data,day):
    before_data = data[(data.day<day)]
    old_user_conv_df = before_data.pivot_table('label','userID',aggfunc='sum')
    old_user_df = pd.DataFrame({'userID':old_user_conv_df.index,'old_user_conv':old_user_conv_df.values})
    #old_user_click_df = before_data.pivot_table('label','userID',aggfunc='count')
    #old_user_click_df = pd.DataFrame({'userID':old_user_click_df.index,'old_user_click':old_user_click_df.values})
    #old_user_df = old_user_df.merge(old_user_click_df,left_on = 'userID',right_on = 'userID',how = 'left')
    return old_user_df


look_back = input('look_back : ')

start_day = data.day.min()+look_back

print 'process day : %d'%start_day
new_data = data[data.day == start_day]
user_df = old_user(data,start_day)
new_data = new_data.merge(user_df,left_on = 'userID',right_on = 'userID',how = 'left').fillna(0)
old_user_df = old_user_conv(data,start_day)
new_data = new_data.merge(old_user_df,left_on = 'userID',right_on = 'userID',how = 'left').fillna(0)

for feature_name in category2value_feature_name:
    s_c_feature = show_conversion_feature(data,start_day,feature_name,look_back)
    new_data = new_data.merge(s_c_feature,left_on = feature_name,right_on = feature_name,how = 'left')


for day in range(start_day+1,data.day.max()+1):
    print 'process day : %d'%day
    day_data = data[data.day == day]
    user_df = old_user(data,day)
    day_data = day_data.merge(user_df,left_on = 'userID',right_on = 'userID',how = 'left').fillna(0)
    old_user_df = old_user_conv(data,day)
    day_data = day_data.merge(old_user_df,left_on = 'userID',right_on = 'userID',how = 'left').fillna(0)
    for feature_name in category2value_feature_name:
        s_c_feature = show_conversion_feature(data,start_day,feature_name,look_back)
        day_data = day_data.merge(s_c_feature,left_on = feature_name,right_on = feature_name,how = 'left')

    new_data = pd.concat([new_data,day_data])

#new_data_copy = new_data.copy()

'''
del_feature = [
 'clickTime',
 'creativeID',
 'userID',
 'positionID',
 'connectionType',
 'telecomsOperator',
 'adID',
 'camgaignID',
 'advertiserID',
 'appID',
 'appPlatform',
 'appCategory',
 'cate',
 'sitesetID',
 'positionType',
 'age',
 'gender',
 'education',
 'marriageStatus',
 'haveBaby',
 'hometown',
 'residence',
 'residence_short',
 'hometown_short',
 'day',
 'click_hour',
 'most_love_have_cate',
 'second_love_have_cate',
 'age_bin',
 'gender_age',
 'positionID_sitesetID',
 'positionType_sitesetID_appID',
 'sitesetID_positionType',
 'positionID_appID',
 'positionID_connectionType',
 'appID_age_bin',
 'appID_gender']

for f in del_feature:
    del new_data[f]
'''

print 'value&category2value_feature shape : ' + str(new_data.shape)

new_data.to_csv('../data/value&category2value_feature.csv',index=False,header=True)



# --------------------------onehot_feature--------------------------

'''

def one_hot_encode(X):
    encoder_one_hot = OneHotEncoder()
    X_one_hot = encoder_one_hot.fit_transform(X.reshape(-1,1))
    return X_one_hot


def one_hot_feature(data,columns_for_onhot):
    feature = np.ones(len(data)).reshape(-1,1)
    for feature_name in columns_for_onhot:
        print feature_name
        onehot = one_hot_encode(data[feature_name])
        feature = np.hstack([feature,onehot.toarray()])
    return feature


columns_for_onhot = [
'connectionType',
'telecomsOperator',
'appPlatform',
'appCategory',
'cate',
'sitesetID',
'positionType',
'gender',
'education',
'marriageStatus',
'click_hour',
'age_bin',
'gender_age',
'sitesetID_positionType',
'positionType_sitesetID_appID',
]

#'positionID_sitesetID',
#'positionID_appID',
#'positionID_connectionType',

onehot_feature = one_hot_feature(new_data_copy,columns_for_onhot)
label = new_data_copy.label


print 'onehot_feature shape : ' + str(onehot_feature.shape)

np.save('../data/onehot_feature.npy',onehot_feature)

np.save('../data/label.npy',label)

'''



