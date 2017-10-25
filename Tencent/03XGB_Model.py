import pandas as pd 
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

#onehot_feature = np.load('../data/onehot_feature.npy')
#label = np.load('../data/label.npy')

value_category2value_feature = pd.read_csv('../data/value&category2value_feature.csv').fillna(0)


import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
'''

lr_model = LogisticRegression(penalty='l2')

#--------

train = onehot_feature[label>=0]
test = onehot_feature[label<0]
train_label = label[label>=0]

lr_model.fit(train,train_label)
pred = lr_model.predict_proba(train)[:,1]
print logloss(train_label,pred)

test_pred = lr_model.predict_proba(test)[:,1]
result_lr = pd.DataFrame({'prob':test_pred,'instanceID':range(1,len(test_pred)+1)})
result_lr.to_csv('submission_lr_onehot.csv',index = False)

#--------

train_value = value_category2value_feature[value_category2value_feature.label>=0].iloc[:,1:].values
test_value = value_category2value_feature[value_category2value_feature.label<0].iloc[:,1:].values
lr_model.fit(train_value,train_label)
pred = lr_model.predict_proba(train_value)[:,1]
print logloss(train_label,pred)


test_pred = lr_model.predict_proba(test_value)[:,1]
result_lr = pd.DataFrame({'prob':test_pred,'instanceID':range(1,len(test_pred)+1)})
result_lr.to_csv('submission_lr_value.csv',index = False)

#---------

'''

'''
feature_important = pd.DataFrame(zip(list(train_feature_norm.columns),list(np.abs(lr_model.coef_[0]))),columns=['feature','score']).sort_values('score',ascending=False)
feature_select = feature_important.head(20).feature.values

lr_model.fit(train_feature_norm[feature_select],train_label)
pred = lr_model.predict_proba(train_feature_norm[feature_select])[:,1]

logloss(train_label.values,pred)


# Need Add one-hot


'''

'''
train_value = value_category2value_feature[value_category2value_feature.label>=0].iloc[:,1:]
test_value = value_category2value_feature[value_category2value_feature.label<0].iloc[:,1:]

def little_xgb(train_value,test_value,train_label):
    param = {
    'eta': 0.2,
    'min_child_weight': 100,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 12,
    'seed': 42,
    'silent': 1
    }
    xg_train = xgb.DMatrix(train_value, label=train_label)
    watchlist = [(xg_train, 'train')]
    model = xgb.train(param, xg_train, verbose_eval=1, evals=watchlist,num_boost_round = 50)
    xgb_feature_importance = model.get_fscore().items()
    xgb_feature_importance.sort(key=lambda x:x[1],reverse=True)
    xg_test = xgb.DMatrix(test_value)
    train_feature_for_xgb = model.predict(xg_train)
    test_feature_for_xgb = model.predict(xg_test)
    print logloss(train_label,train_feature_for_xgb)
    return train_feature_for_xgb,test_feature_for_xgb,xgb_feature_importance

train_feature_for_xgb,test_feature_for_xgb,xgb_feature_importance = little_xgb(train_value,test_value,train_label)


#-------------------------------


lr_model.fit(np.hstack([train,train_feature_for_xgb.reshape(-1,1)]),train_label)
pred = lr_model.predict_proba(np.hstack([train,train_feature_for_xgb.reshape(-1,1)]))[:,1]
print logloss(train_label,pred)

test_pred = lr_model.predict_proba(np.hstack([test,test_feature_for_xgb.reshape(-1,1)]))[:,1]
result_lr = pd.DataFrame({'prob':test_pred,'instanceID':range(1,len(test_pred)+1)})
result_lr.to_csv('submission_lr_with_xgb.csv',index = False)


# -----------------------------

importanct_feature_select = [n[0] for n in xgb_feature_importance][:10]

train_value_select = value_category2value_feature[value_category2value_feature.label>=0].loc[:,importanct_feature_select].values
test_value_select = value_category2value_feature[value_category2value_feature.label<0].loc[:,importanct_feature_select].values

lr_model.fit(np.hstack([train,train_value_select]),train_label)
pred = lr_model.predict_proba(np.hstack([train,train_value_select]))[:,1]
print logloss(train_label,pred)

test_pred = lr_model.predict_proba(np.hstack([test,test_value_select]))[:,1]
result_lr = pd.DataFrame({'prob':test_pred,'instanceID':range(1,len(test_pred)+1)})
result_lr.to_csv('submission_lr_select_feature.csv',index = False)

'''
#----xgb-----


def little_xgb(train_value,train_label):
    param = {
    'eta': 0.2,
    'min_child_weight': 100,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 12,
    'seed': 42,
    'silent': 1
    }
    xg_train = xgb.DMatrix(train_value, label=train_label)
    watchlist = [(xg_train, 'train')]
    model = xgb.train(param, xg_train, verbose_eval=1, evals=watchlist,num_boost_round = 50)
    xgb_feature_importance = model.get_fscore().items()
    xgb_feature_importance.sort(key=lambda x:x[1],reverse=True)
    return xgb_feature_importance








train_data = value_category2value_feature[value_category2value_feature.label>=0].reset_index(drop=1)

test_feature = value_category2value_feature[value_category2value_feature.label==-1].reset_index(drop=1).iloc[:,1:]
#xgb_feature_importance = little_xgb(train_data.iloc[:,1:],train_data.label)
#feature_select = ['label','day']+[x[0] for x in xgb_feature_importance if x[1]>10]

#test_feature = value_category2value_feature[value_category2value_feature.label==-1].reset_index(drop=1)[feature_select].iloc[:,1:]
#train_data = train_data[feature_select]


#eta = input('eta: ')
num_round = input('round: ')

param_grid = [
    {
    'eta': [0.2],
    'gamma': [0.0],
    'max_depth': [7],
    'min_child_weight': [100],
    'max_delta_step': [0],
    'subsample': [1],
    'colsample_bytree': [0.6],
    'colsample_bylevel': [1],
    'lambda': [1],
    'alpha': [0],
    'tree_method': ['approx'],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss'],
    'nthread': [12],
    'seed': [42],
    'silent': [1]
    }
]

#num_round = 150 
early_stop = 5
best_score = 10
best_params = None



train_data_train = train_data[train_data.day<29]
train_data_val = train_data[train_data.day==29]

train_data_val_feature = train_data_val.iloc[:,1:]
train_data_val_label = train_data_val.label.values

sample_times = input('sample times : ')

dval = xgb.DMatrix(train_data_val_feature)
dval_watch = xgb.DMatrix(train_data_val_feature, label=train_data_val_label)
for xgb_pars in ParameterGrid(param_grid):
    print xgb_pars
    pred_list = []
    for i in range(sample_times):
        train_data_train_sample = train_data_train.sample(int(len(train_data_train)*0.9))
        train_data_train_sample_feature = train_data_train_sample.iloc[:,1:]
        train_data_train_sqmple_label = train_data_train_sample.label
        dtrain = xgb.DMatrix(train_data_train_sample_feature, label=train_data_train_sqmple_label)
        watchlist = [(dtrain, 'train'),((dval_watch, 'dval_watch'))]
        model = xgb.train(xgb_pars, dtrain, verbose_eval=1, evals=watchlist,num_boost_round = num_round,early_stopping_rounds = early_stop)
        pred = model.predict(dval)
        pred_list.append(pred)
        print logloss(train_data_val_label,pred)
    pred= np.array(pred_list).mean(0)
    loss = logloss(train_data_val_label,pred)
    print loss
    if loss < best_score:
        best_params = xgb_pars
        best_score = loss



best_iter = input('best_iter: ')

train_data = train_data[train_data.day!=30]

dtest = xgb.DMatrix(test_feature)
pred_list = []
for i in range(sample_times):

    train_data_sample = train_data.sample(int(len(train_data)*0.9))

    train_data_sample_feature = train_data_sample.iloc[:,1:]
    train_data_sqmple_label = train_data_sample.label

    dtrain = xgb.DMatrix(train_data_sample_feature, label=train_data_sqmple_label)
    watchlist = [(dtrain, 'train')]
    model = xgb.train(best_params, dtrain, verbose_eval=1, evals=watchlist,num_boost_round = best_iter,early_stopping_rounds = 10)
    xgb_feature_importance = model.get_fscore().items()
    xgb_feature_importance.sort(key=lambda x:x[1],reverse=True)
    print xgb_feature_importance[:10]
    pred = model.predict(dtest)
    pred_list.append(pred)

pred= np.array(pred_list).mean(0)


result = pd.DataFrame({'prob':pred,'instanceID':range(1,len(pred)+1)})

test_InstanceID = pd.read_csv('test_InstanceID.csv',names = ['instanceID'])

result.instanceID = test_InstanceID.instanceID.values
result = result.sort_values('instanceID')

result.to_csv('submission.csv',index = False)


