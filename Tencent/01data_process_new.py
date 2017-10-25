import pandas as pd

print 'read data'

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
ad = pd.read_csv('../data/ad.csv')
app_categories = pd.read_csv('../data/app_categories.csv')
position = pd.read_csv('../data/position.csv')
user = pd.read_csv('../data/user.csv')
user_app_actions = pd.read_csv('../data/user_app_actions.csv')
user_installedapps = pd.read_csv('../data/user_installedapps.csv')



print 'process data'


print 'merge '

app_categories.appCategory = app_categories.appCategory.astype('str')
app_categories['cate'] = app_categories.appCategory.apply(lambda x: x[0])


user['residence_short'] = user['residence'].apply(lambda x:int(x/100))
user['hometown_short'] = user['hometown'].apply(lambda x:int(x/100))

train = train.merge(ad,left_on='creativeID',right_on='creativeID',how='left')
test = test.merge(ad,left_on='creativeID',right_on='creativeID',how='left')

train = train.merge(app_categories,left_on='appID',right_on='appID',how='left')
test = test.merge(app_categories,left_on='appID',right_on='appID',how='left')

train = train.merge(position,left_on='positionID',right_on='positionID',how='left')
test = test.merge(position,left_on='positionID',right_on='positionID',how='left')

train = train.merge(user,left_on='userID',right_on='userID',how='left')
test = test.merge(user,left_on='userID',right_on='userID',how='left')

train['day'] = train.clickTime.apply(lambda x: int(x/10000))
test['day'] = test.clickTime.apply(lambda x: int(x/10000))

train['click_hour'] = train.clickTime.apply(lambda x: int(x%10000/100))
test['click_hour'] = test.clickTime.apply(lambda x: int(x%10000/100))

user_app_actions = user_app_actions.merge(app_categories,left_on='appID',right_on='appID',how='left')
user_installedapps = user_installedapps.merge(app_categories,left_on='appID',right_on='appID',how='left')
user_have_cate = user_installedapps.pivot_table('appID','userID','cate',aggfunc='count').fillna(0)

#--------------------

print 'user feature '

def find_second_max(M):
    return pd.DataFrame((M - M.max(1).reshape(-1,1))).replace(0,-100000000).values.argmax(1)
def find_most_max(M):
    return M.argmax(1)

users =  pd.concat([train.userID,test.userID]).drop_duplicates()

user_install_cate = user_installedapps.pivot_table('appID','userID','cate',aggfunc='count').fillna(0)
user_feature_more = pd.DataFrame()
user_feature_more['userID'] = users.values

most_love_have_cate_df = pd.DataFrame({'userID':user_have_cate.index ,'most_love_have_cate':find_most_max(user_have_cate.values)})
second_love_have_cate_df = pd.DataFrame({'userID':user_have_cate.index ,'second_love_have_cate':find_second_max(user_have_cate.values)})

user_feature_more = user_feature_more.merge(most_love_have_cate_df,left_on='userID',right_on='userID',how='outer').fillna(-1)
user_feature_more = user_feature_more.merge(second_love_have_cate_df,left_on='userID',right_on='userID',how='outer').fillna(-1)
#user_feature_more = user_feature_more.where(~np.isnan(user_feature_more),user_feature_more.mode(),axis=1).astype('int') # fillna with mode
user_have_cate['userID'] = user_have_cate.index
user_have_cate.columns = ['cate0_num','cate1_num','cate2_num','cate3_num','cate4_num','cate5_num','userID']
user_feature_more = user_feature_more.merge(user_have_cate,left_on='userID',right_on='userID',how='outer').fillna(0)

train = train.merge(user_feature_more,left_on='userID',right_on='userID',how='left').fillna(-1)
test = test.merge(user_feature_more,left_on='userID',right_on='userID',how='left').fillna(-1)

train['most_love_have_cate_diff'] = [1 if x == 0 else 0 for x in (train['most_love_have_cate'].astype('int') - train['cate'].astype('int'))]
train['second_love_have_cate_diff'] = [1 if x == 0 else 0 for x in (train['second_love_have_cate'].astype('int') - train['cate'].astype('int'))]
test['most_love_have_cate_diff'] = [1 if x == 0 else 0 for x in (test['most_love_have_cate'].astype('int') - test['cate'].astype('int'))]
test['second_love_have_cate_diff'] = [1 if x == 0 else 0 for x in (test['second_love_have_cate'].astype('int') - test['cate'].astype('int'))]
#-----------------------

print 'interact feature '
def age_bin(age):
    return int(age/10)

train['age_bin'] = train.age.apply(age_bin).astype('str')
train['gender'] = train.gender.astype('str')
train['gender_age'] = train.gender + train.age_bin
train['positionID_sitesetID'] = train.positionID.astype('str')+train.sitesetID.astype('str')
train['positionType_sitesetID_appID'] = train.positionType.astype('str')+train.sitesetID.astype('str') + train.appID.astype('str')
train['sitesetID_positionType'] = train.sitesetID.astype('str')+train.positionType.astype('str')
train['positionID_appID'] = train.positionID.astype('str')+train.appID.astype('str')
train['positionID_connectionType'] = train.positionID.astype('str')+train.connectionType.astype('str')
train['appID_age_bin'] = train.appID.astype('str')+train.age_bin.astype('str')
train['appID_gender'] = train.appID.astype('str')+train.gender.astype('str')

test['age_bin'] = test.age.apply(age_bin).astype('str')
test['gender'] = test.gender.astype('str')
test['gender_age'] = test.gender + test.age_bin
test['positionID_sitesetID'] = test.positionID.astype('str')+test.sitesetID.astype('str')
test['positionType_sitesetID_appID'] = test.positionType.astype('str')+test.sitesetID.astype('str') + test.appID.astype('str')
test['sitesetID_positionType'] = test.sitesetID.astype('str')+test.positionType.astype('str')
test['positionID_appID'] = test.positionID.astype('str')+test.appID.astype('str')
test['positionID_connectionType'] = test.positionID.astype('str')+test.connectionType.astype('str')
test['appID_age_bin'] = test.appID.astype('str')+test.age_bin.astype('str')
test['appID_gender'] = test.appID.astype('str')+test.gender.astype('str')

#----------------------------------------------------------

def get_day_hour(s):
    return s[:4]


train['num'] = range(len(train))
train['same'] = train['clickTime'].astype('str').apply(get_day_hour) +  '-'+ train['creativeID'].astype('str') +  '-'+ train['userID'].astype('str') +  '-'+ train['appID'].astype('str')
train = train.sort_values(['same','num'],ascending=True)
rank = [1]
s = train.same[0]
for x in train.same[1:]:
    if x == s:
        rank.append(rank[-1]+1)
    else:
        rank.append(1)
    s = x
train['rank_'] = rank

train_same_click_count = train.pivot_table('appID','same',aggfunc = 'count')
train_same_click_count_df  = pd.DataFrame({'same':train_same_click_count.index ,'same_click_count':train_same_click_count.values})
train = train.merge(train_same_click_count_df,left_on='same',right_on='same',how='left')


test['num'] = range(len(test))
test['same'] = test['clickTime'].astype('str').apply(get_day_hour) + '-'+ test['creativeID'].astype('str') + '-'+ test['userID'].astype('str') + '-'+ test['appID'].astype('str')
test = test.sort_values(['same','num'],ascending=True)
rank = [1]
s = test.same[0]
for x in test.same[1:]:
    if x == s:
        rank.append(rank[-1]+1)
    else:
        rank.append(1)
    s = x

test['rank_'] = rank

test_same_click_count = test.pivot_table('appID','same',aggfunc = 'count')
test_same_click_count_df  = pd.DataFrame({'same':test_same_click_count.index ,'same_click_count':test_same_click_count.values})
test = test.merge(test_same_click_count_df,left_on='same',right_on='same',how='left')

del train['same']
del test['same']

#----------------------------------------------------------

print 'userID_day'

train['userID_day'] = train['userID'].astype('str') + '-' + train['day'].astype('str')
train_user_day_click = train.pivot_table('clickTime','userID_day',aggfunc='count')
train_user_day_click_df  = pd.DataFrame({'userID_day':train_user_day_click.index ,'user_day_click':train_user_day_click.values})
train = train.merge(train_user_day_click_df,left_on='userID_day',right_on='userID_day',how='left')
del train['userID_day']

test['userID_day'] = test['userID'].astype('str') + '-' + test['day'].astype('str')
test_user_day_click = test.pivot_table('clickTime','userID_day',aggfunc='count')
test_user_day_click_df  = pd.DataFrame({'userID_day':test_user_day_click.index ,'user_day_click':test_user_day_click.values})
test = test.merge(test_user_day_click_df,left_on='userID_day',right_on='userID_day',how='left')
del test['userID_day']



#-----------------------------------------------------



print 'write files'

train.to_csv('../data/train_new.csv',header=True,index=False)
test.to_csv('../data/test_new.csv',header=True,index=False)
user_app_actions.to_csv('../data/user_app_actions_new.csv',header=True,index=False)
user_installedapps.to_csv('../data/user_installedapps_new.csv',header=True,index=False)


