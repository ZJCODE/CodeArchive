# -*- coding: utf-8 -*-
'''
输入：
    模型文件名
    样本文件
输出：
    模型文件
    LR auc
    GBDT auc
    GBDT leaf -> LR auc
    GBDT leaf + RAW -> LR auc
'''

import xgboost as xgb
import numpy as np
import sys
import re
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import cPickle
import pandas as pd

#计算缩进
def count_indent(line):
    count = 0
    for char in line:
        if char != '\t':
            break
        count += 1
    return count

#获取本节点用第几个feature做分支的
def get_feature(line):
    pattern = re.compile(r'f.*<')
    match = pattern.search(line)
    return int(match.group()[1:-1])

#获取本节点做分支的阀值
def get_threshold(line):
    pattern = re.compile(r'<.*]')
    match = pattern.search(line)
    return float(match.group()[1:-1])

#获取本节点特征确实的分支方向
def get_miss(line1, line2):
    pattern = re.compile(r'missing=.*')
    match = pattern.search(line1)
    miss = match.group()[8:]
    pattern = re.compile(r'\t.*:')
    match = pattern.search(line2)
    left = match.group()[count_indent(line2):-1]
    if miss == left:
        return "left"
    else:
        return "right"

#获取叶子节点的编号
def get_leaf(line):
    pattern = re.compile(r'\t.*:')
    match = pattern.search(line)
    return match.group()[count_indent(line):-1]

#训练gbdt，之后导出模型文件
def train_gbdt(data_file, train_set, eval_set, params):
    dtrain = xgb.DMatrix(train_set[:, 1:], train_set[:, 0])
    dtest = xgb.DMatrix(eval_set[:, 1:], eval_set[:, 0])
    param = {'max_depth':params['max_depth'], 'eta':1, 'silent':1, 'objective':'binary:logistic' ,'eval_metric':'auc'}
    watchlist  = [(dtest,'eval'), (dtrain,'train')]
    print "GBDT RESULT:"
    bst = xgb.train(param, dtrain, params['num_round'], watchlist)
    #print pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
    bst.dump_model(model_file)

'''
用于存储GBDT model的类
    init方法从模型文件中读入模型在内存中存储成树形结构
    predict方法把一组raw feature转换成GBDT leaf.
'''
class GbdtModel:
    def __init__(self, model_file):
        self.trees = []
        file_content = []
        tree_no = 0
        for line in open(model_file).readlines():
            if line[0:7] == 'booster':
                if len(file_content) != 0:
                    self.trees.append(TreeNode(file_content, tree_no))
                    tree_no += 1
                file_content = []
            else:
                file_content.append(line.rstrip())
        self.trees.append(TreeNode(file_content, tree_no))
        global feature_no
        self.feature_no = feature_no

    def predict(self, data_set):
        features_leaf = []
        labels = []
        for data in data_set:
            features_leaf.append(self.predict_one(data[1:]))
            labels.append(int(data[0]))
        return np.array(features_leaf), np.array(labels)

    def predict_one(self, feature_raw):
        feature = np.zeros(self.feature_no).astype(int).tolist()
        for tree in self.trees:
            feature[tree.predict(feature_raw)] = 1
        return feature
        
#树节点类，递归的建树，递归的predict
class TreeNode:
    def __init__(self, file_content, tree_no):
        self.feature = get_feature(file_content[0])
        self.threshold = get_threshold(file_content[0])
        self.miss = get_miss(file_content[0], file_content[1])
        indent = count_indent(file_content[1])
        child_content = [file_content[1]]
        for line in file_content[2:]:
            if count_indent(line) == indent:
                if len(child_content) == 1:
                    self.left = Leaf(child_content[0], tree_no)
                else:
                    self.left = TreeNode(child_content, tree_no)
                child_content = [line]
            else:
                child_content.append(line.rstrip())
        if len(child_content) == 1:
            self.right = Leaf(child_content[0], tree_no)
        else:
            self.right = TreeNode(child_content, tree_no)

    def predict(self, feature):
        if float(feature[self.feature]) < self.threshold:
            return self.left.predict(feature)
        else:
            return self.right.predict(feature)

#叶子节点类，其中tree_no只是做为tune模型的时候看模型中间结果用
class Leaf:
    def __init__(self, content, tree_no):
        global feature_no
        self.tree_no = str(tree_no)+"_"+get_leaf(content)
        self.feature_no = feature_no
        feature_no += 1

    def predict(self, feature):
        return self.feature_no

if __name__ == '__main__':
    model_file = sys.argv[1]
    data_file = sys.argv[2]
    #data_file2 = sys.argv[3]
    #GBDT参数
    params = {
        'max_depth':3,
        'num_round':10,
    }
    data = np.loadtxt(data_file)
    #data2 = np.loadtxt(data_file2)
    np.random.shuffle(data)
    #train_set = data
    #eval_set = data2
    train_set = data[1000:]
    eval_set = data[:1000]
    train_gbdt(data_file, train_set, eval_set, params)
    feature_no = 0
    model = GbdtModel(model_file)
    print "GBDT leaf count: "+str(feature_no)
    train_leaf, train_label = model.predict(train_set)
    eval_leaf, eval_label = model.predict(eval_set)

    fold_count = 5
    KF = cross_validation.KFold(len(train_label), n_folds=fold_count, shuffle=False)
    auc = 0
    print "GBDT_LEAF->LR RESULT(feature_count="+str(len(train_leaf[0]))+"):"
    for train, test in KF:
        clf_l2_LR = LogisticRegression(tol=0.0001)
        print train_leaf[train].shape
        clf_l2_LR.fit(train_leaf[train], train_label[train])
        y_prob = clf_l2_LR.predict_proba(train_leaf[test])
        fold_auc = metrics.roc_auc_score(train_label[test],y_prob[:,1],average='macro', sample_weight=None)
        auc += fold_auc
        print "fold_auc: "+str(fold_auc)
    print "CV_auc: "+str(auc/fold_count)
    y_validation = clf_l2_LR.predict_proba(eval_leaf)
    #for i in range(0,len(y_validation[:,1])):
    #    print str(eval_label[i])+"\t"+str(y_validation[:,1][i])
    validation_auc = metrics.roc_auc_score(eval_label,y_validation[:,1],average='macro', sample_weight=None)
    print "validation_auc"+str(validation_auc)

    train_leaf_raw = np.append(train_leaf, train_set[:,1:], axis = 1)
    eval_leaf_raw = np.append(eval_leaf, eval_set[:,1:], axis = 1)
    scaler = StandardScaler().fit(train_leaf_raw)
    train_leaf_raw = scaler.transform(train_leaf_raw)
    eval_leaf_raw = scaler.transform(eval_leaf_raw)

    auc = 0
    print "GBDT_LEAF_RAW->LR RESULT(feature_count="+str(len(train_leaf_raw[0]))+"):"
    for train, test in KF:
        clf_l2_LR = LogisticRegression(C=0.1, penalty='l2',tol=0.0001)
        clf_l2_LR.fit(train_leaf_raw[train], train_label[train])
        y_prob = clf_l2_LR.predict_proba(train_leaf_raw[test])
        fold_auc = metrics.roc_auc_score(train_label[test],y_prob[:,1],average='macro', sample_weight=None)
        auc += fold_auc
        print "fold_auc: "+str(fold_auc)
    print "CV_auc: "+str(auc/fold_count)
    y_validation = clf_l2_LR.predict_proba(eval_leaf_raw)
    validation_auc = metrics.roc_auc_score(eval_label,y_validation[:,1],average='macro', sample_weight=None)
    print "validation_auc"+str(validation_auc)

    train_raw = train_set[:,1:]
    eval_raw = eval_set[:,1:]
    scaler = StandardScaler().fit(train_raw)
    train_raw = scaler.transform(train_raw)
    eval_raw = scaler.transform(eval_raw)

    auc = 0
    print "RAW->LR RESULT(feature_count="+str(len(train_raw[0]))+"):"
    for train, test in KF:
        clf_l2_LR = LogisticRegression(C=1000, penalty='l2',tol=0.0001)
        clf_l2_LR.fit(train_raw[train], train_label[train])
        y_prob = clf_l2_LR.predict_proba(train_raw[test])
        fold_auc = metrics.roc_auc_score(train_label[test],y_prob[:,1],average='macro', sample_weight=None)
        auc += fold_auc
        print "fold_auc: "+str(fold_auc)
    print "CV_auc: "+str(auc/fold_count)
    y_validation = clf_l2_LR.predict_proba(eval_raw)
    validation_auc = metrics.roc_auc_score(eval_label,y_validation[:,1],average='macro', sample_weight=None)
    print "validation_auc"+str(validation_auc)
