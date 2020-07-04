#!/usr/bin/python -u
# -*- encoding:utf-8 -*-

'''
   主办方将这样调用你的程序以便验证
	   python main.py --data_dir  /data1/studio_round1/ --res_file /data1/studio_round1_result/someone/studio.txt
	参数
		data_dir : 数据所在目录, 此目录下有role_create,role_login,role_logout等数据
		res_file : 最终判定结果将写入此文件
		
'''

import sys
import os
import time
import argparse
import gc

import numpy as np
import pandas as pd
try:
	import lightgbm as lgb
except:
	os.system("pip install lightgbm")
	import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from get_features import make

# 处理数据, 训练模型, 预测等
# 具体逻辑可以写到其他文件中, 在这里统一调用
def do_sth(data_dir, res_file):
	print(data_dir, res_file)
	make(data_dir)
	model(res_file)
	
def fscore(y_true, y_pred, threshhold = 0.5):
	y_pred = (y_pred > threshhold) * 1
	R = recall_score(y_true, y_pred)
	P = precision_score(y_true, y_pred)
	return 4*P*R / (P+3*R)

def model(res_file):
	train_df = pd.read_csv("./dataset/train.csv")
	test_df = pd.read_csv("./dataset/test.csv")
	
	money_tr =pd.read_csv("./dataset/train_money.csv")
	money_te =pd.read_csv("./dataset/test_money.csv")

	fec = money_tr.columns.tolist()[1:]

	train_df = train_df.merge(money_tr, how="inner", on="uin").sort_values("uin")
	test_df = test_df.merge(money_te, how="inner", on="uin")
	
	for c in ["onlinetime_sum", "onlinetime_mean", "onlinetime_max", "onlinetime_std"]:
		train_df.loc[train_df.loginseconds_mean.notna(), c] = train_df.loc[train_df.loginseconds_mean.notna(), c.replace("onlinetime", "loginseconds")]
		test_df.loc[test_df.loginseconds_mean.notna(), c] = test_df.loc[test_df.loginseconds_mean.notna(), c.replace("onlinetime", "loginseconds")]
		
	train_features = ['clientip_nunique', 'network_nunique',\
					  'onlinetime_sum', 'onlinetime_mean', 'onlinetime_count', 'onlinetime_max', 'onlinetime_std', #, 'no_logout'\
					  'net_2G/3G', 'net_4G', 'net_WIFI', #, 'net_LTE', 'net_LTE/GSM', 'net_UNKNOWN'\
					  'friendsadd_sum', 'friendsadd_mean', 'friendsadd_max', 'friendsadd_std', \
					  'friendsnum_max_sum', 'friendsnum_max_mean', 'friendsnum_max_max', 'friendsnum_max_std', \
					  'poweradd_sum', 'poweradd_mean', 'poweradd_max', 'poweradd_std', \
					  'power_max_sum', 'power_max_mean', 'power_max_max', 'power_max_std', \
					  'roleleveladd_sum', 'roleleveladd_mean', 'roleleveladd_max', 'roleleveladd_std', \
					  'rolelevel_max_sum', 'rolelevel_max_mean', 'rolelevel_max_max', 'rolelevel_max_std', \
					  'job_1', 'job_2', 'job_3', 'job_4', \
					  'login_len', 'login_dawn_hour_num',\
					  "loginseconds_sum", "loginseconds_mean", "loginseconds_max", "loginseconds_std",\
					  'is_createtoday', 'create_hour', 'chat_cnt']+fec
	print(train_features, len(train_features))
	
	params = {
		"objective": "binary",
		'num_leaves': 63,
		"learning_rate": 0.05,
		"lambda_l1": 0.1,
		"lambda_l2": 0.1,
		'feature_fraction': 0.75,
		'bagging_fraction': 0.9,
		'bagging_freq': 3,
		'seed': 2050,
		"iterations": 6666,
		"num_threads ": 16,
		'metric': {'binary_logloss', 'auc'},
		'is_unbalance': True
	}
	pred_tests = []
	pred_vals = []
	true_vals = []

	skf = StratifiedKFold(n_splits=5, random_state=2050, shuffle=False)
	for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
		# print('fold:', fold)
		train_data_list = train_df.iloc[train_idx]
		val_data_list = train_df.iloc[val_idx]
		lgb_train = lgb.Dataset(train_data_list[train_features].values, train_data_list['target'].values)
		lgb_eval = lgb.Dataset(val_data_list[train_features].values, val_data_list['target'].values, reference=lgb_train)
		gbm = lgb.train(params,
					lgb_train,
					num_boost_round=2333,
					valid_sets=[lgb_train, lgb_eval],
					valid_names=["train",'valid'],
					feature_name=train_features,
					early_stopping_rounds=300,
	#                 feval=self_metric,
					verbose_eval=50,)
		pred_val = gbm.predict(val_data_list[train_features].values, num_iteration=gbm.best_iteration)
		# print("acc:",accuracy_score(val_data_list['target'].values, np.round(pred_val)))
		# print("fscore", fscore(val_data_list['target'].values, pred_val))
		# print("fscore", fscore(val_data_list['target'].values, pred_val, threshhold=0.85))
		print("fscore", fscore(val_data_list['target'].values, pred_val, threshhold=0.875))
		pred_vals.append(pred_val)
		true_vals.append(val_data_list['target'].values)
		pred_test = gbm.predict(test_df[train_features].values, num_iteration=gbm.best_iteration)
		pred_tests.append(pred_test)
		print(np.array(pred_tests).shape, pred_test.shape)
	y_pred = np.hstack(pred_vals)
	y_true = np.hstack(true_vals)
	# print("fscore", fscore(y_true, y_pred ))
	# print("fscore", fscore(y_true, y_pred, 0.85))
	# print("fscore", fscore(y_true, y_pred, 0.875))
	pred_tests = np.array(pred_tests)
	result = np.mean(pred_tests, 0)
	test_df["target"] = 1*(result>0.875)
	test_df.loc[test_df.target == 1, "uin"].drop_duplicates().to_csv(res_file, sep="\n", index=False)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",    type=str, required=True,  help='e.g. /data1/studio_round1/ ')
	parser.add_argument("--res_file",    type=str, required=True,  help='e.g. /data1/studio_round1_result/someone/studio.txt')
	args = parser.parse_args()
	
	do_sth(args.data_dir, args.res_file)
	
