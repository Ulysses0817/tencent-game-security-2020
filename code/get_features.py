#!/usr/bin/env python
# coding: utf-8

import os
import gc
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

def load_loginout(mode='train', labels=None, data_dir=None):
	"""
	读取登陆登出信息
	"""
	cols = ["dteventtime", "platid", "areaid", "worldid", "uin", "roleid", "rolename",\
			"job", "rolelevel", "power", "friendsnum", "network", "clientip", "deviceid"]

	if mode=="train":
		file_name = "20200301.txt"
	elif mode=="test":
		file_name = "20200305.txt"

	role_login = pd.read_csv(os.path.join(data_dir, "role_login/", file_name), delimiter="|", names=cols)
	role_logout = pd.read_csv(os.path.join(data_dir, "role_logout/", file_name), delimiter="|", names=cols+["onlinetime"])

	if labels is not None:
		role_login = role_login.loc[role_login.uin.isin(labels)]
		role_logout = role_logout.loc[role_logout.uin.isin(labels)]
	
	role_login.dteventtime = pd.to_datetime(role_login.dteventtime)
	role_logout.dteventtime = pd.to_datetime(role_logout.dteventtime)
	
	return role_login.sort_values('dteventtime').reset_index(drop=True), role_logout.sort_values('dteventtime').reset_index(drop=True)


def load_create_chat(mode='train', labels=None, data_dir=None):
	"""
	读取账号注册与聊天信息
	"""
	cols = ["dteventtime", "platid", "areaid", "worldid", "uin", "roleid", "rolename",\
			"job", "rolechannel", "network", "clientip", "deviceid"]

	if mode=="train":
		file_name = "20200301.txt"
	elif mode=="test":
		file_name = "20200305.txt"
		
	role_create = pd.read_csv(os.path.join(data_dir, "role_create/"+file_name), delimiter="|", names=cols)
	role_chat = pd.read_csv(os.path.join(data_dir, "uin_chat/"+file_name), delimiter="|", names=["uin", "chat_cnt"])

	if labels is not None:
		role_create = role_create.loc[role_create.uin.isin(labels)]
		role_chat = role_chat.loc[role_chat.uin.isin(labels)]

	role_create.dteventtime = pd.to_datetime(role_create.dteventtime)
	
	return role_create.sort_values('dteventtime').reset_index(drop=True), role_chat

def load_money_item(mode='train', labels=None, data_dir=None):
	"""
	读取账号注册与聊天信息
	"""
	money_cols = ["dteventtime", "worldid", "uin", "roleid", "rolelevel",\
			"iMoneyType", "iMoney", "AfterMoney", "AddOrReduce", "Reason", "SubReason"]
	item_cols = ["dteventtime", "worldid", "uin", "roleid", "rolelevel",\
			"Itemtype", "Itemid", "Count", "Aftercount", "AddOrReduce", "Reason", "SubReason"]
	
	# 防止内存溢出
	MONEY_TYPEs={"worldid": "int32", "rolelevel": "int16", "iMoneyType":"int16", "iMoney":"int32", "AfterMoney":"int32", \
			 "AddOrReduce":"int8", "Reason":"int16", "SubReason":"int16"}
	ITEM_TYPEs={"worldid": "int32", "rolelevel": "int16", "ItemType":"int16", "Count":"int32", "Aftercount":"int32",\
				"AddOrReduce":"int8", "Reason":"int16", "SubReason":"int16"}
	
	if mode=="train":
		file_name = "20200301.txt"
	elif mode=="test":
		file_name = "20200305.txt"
		
	role_money = pd.read_csv(os.path.join(data_dir, "role_moneyflow/"+file_name), delimiter="|", names=money_cols, dtype=MONEY_TYPEs)
	role_item = pd.read_csv(os.path.join(data_dir, "role_itemflow/"+file_name), delimiter="|", names=item_cols, dtype=ITEM_TYPEs)

	if labels is not None:
		role_money = role_money.loc[role_money.uin.isin(labels)]
		role_item = role_item.loc[role_item.uin.isin(labels)]
	
	role_money.dteventtime = pd.to_datetime(role_money.dteventtime)
	role_item.dteventtime = pd.to_datetime(role_item.dteventtime)
	
	return role_money.sort_values('dteventtime').reset_index(drop=True), role_item.sort_values('dteventtime').reset_index(drop=True)#

def dropuna(df):
	"""
	drop unqiue value columns and rows with Nan values
	"""
	# drop nan
	df.dropna(axis=1, how="all", inplace=True)
	df.dropna(inplace=True)
	# drop unique
	drop_cols = (df.nunique() == 1)
	drop_cols = drop_cols.loc[drop_cols == 1].index.tolist()
	print("drop column:{}".format(drop_cols))
	df.drop(columns=drop_cols, inplace=True)
	return df

def printuninum(dfs):
	for _c in dfs[0].columns:
		for df in dfs:
			print("{}：{}".format(_c, len(df[_c].unique())), end="\t")
		print("")
	print("-"*60)

def concat_df(role_login_tr, role_logout_tr):
	role_login_tr["is_login"] = 1
	role_logout_tr["is_login"] = 0
	return pd.concat([role_login_tr, role_logout_tr], ignore_index=True)


# ## Feature

# #### uin粒度的特征
# - 登陆次数
# - 登出次数
# - 角色个数
# - 在线总时间
# - 同时在线角色个数
# - 最早上线小时点
# - 

# ip特征 
# - uin对应ip个数
# - uin-roleid对应ip个数的统计特征
# - 对ip进行count编码/target编码 https://blog.csdn.net/ssswill/article/details/90271293
# 
# 时间特征：
# - 在线总时长
# - 最早上线时间点
# - 最晚上线时间点
# 
# 属性特征：
# - 角色个数
# - 角色职业的纯度
# - 朋友总数、朋友数量变化
# - 等级变化
# - power变化

def dawn_hour_num(se):
	return (se<"2020-03-01 06:30:00").sum()

def unique_len(se):
	return len(np.unique(se))

def std(array):
	return np.std(array)

def login_fe(df):
	features_df = df[["uin"]].drop_duplicates().reset_index(drop=True)
	df_groupby_uin = df.groupby(["uin"])
	del df
	gc.collect()
	
	# login的次数
	login_len = df_groupby_uin.agg({"job":[len]}).reset_index()
	login_len.columns = ['uin', 'login_len']
	features_df = pd.merge(features_df, login_len, how='left', on='uin')
	
	# 06：30前login的次数
	login_dawn_hour_num = df_groupby_uin.agg({"dteventtime": [dawn_hour_num]}).reset_index()
	login_dawn_hour_num.columns = ['uin', 'login_dawn_hour_num']
	features_df = pd.merge(features_df, login_dawn_hour_num, how='left', on=['uin'])
	
	return features_df

def inout_fe(role_inout, mode="train"):
	features_df = role_inout[["uin"]].drop_duplicates().reset_index(drop=True)
	
	# 1. groupby(["uin"])
	role_inout_groupby_uin = role_inout.groupby(["uin"])
	
	# unique clientip & net features
	df = role_inout_groupby_uin[["clientip", "network", "roleid"]].nunique().reset_index()
	df.columns = ['uin', 'clientip'+'_nunique', 'network'+'_nunique', "roleid_nunique"]
	features_df = pd.merge(features_df, df, how='left', on='uin')
	
	# onlinetime features
	df = role_inout_groupby_uin.agg({"onlinetime":["sum", "mean", "count", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	df.loc[df.onlinetime_std.isna(), "no_logout"] = 1
	NA_uins = df.loc[df.onlinetime_std.isna(), "uin"].values
	df.fillna(0, inplace=True)
	features_df = pd.merge(features_df, df, how='left', on='uin')
	
	# calculate onlinetime of no-logout uins
	ENDTIME = pd.datetime(2020, 3, 2) if mode == "train" else pd.datetime(2020, 3, 6)
	role_inout.loc[role_inout.uin.isin(NA_uins), "loginseconds"] = role_inout.loc[role_inout.uin.isin(NA_uins), "dteventtime"].apply(lambda x: (ENDTIME - x).seconds)
	df = role_inout.groupby(["uin", "roleid"])["loginseconds"].median().reset_index().groupby(["uin"]).agg({"loginseconds":["sum", "mean", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	features_df = pd.merge(features_df, df, how='left', on='uin')
	role_inout.drop(columns="loginseconds", inplace=True)
	
	# 2. groupby(["uin", "roleid"])
	role_inout_group = role_inout.groupby(["uin", "roleid"])
	
	# net features one hot
	role_inout.network.replace(["2G", "3G"], value="2G/3G", inplace=True)
	role_inout.network.replace(["LTE_CA", "GSM"], value='LTE/GSM', inplace=True)
	role_inout.network.replace("unknown", value='UNKNOWN', inplace=True)
	df = role_inout.loc[:, ["uin", "roleid"]].join(pd.get_dummies(role_inout.network, prefix="net"))
	df = df.groupby("uin").max().reset_index().drop(columns=["roleid"])
	features_df = pd.merge(features_df, df, how='left', on='uin')
	
	# friend features
	role_inout_group_friend = role_inout_group.agg({"friendsnum":[min, max]}).reset_index()
	role_inout_group_friend.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in role_inout_group_friend.columns ]

	role_inout_group_friend["friendsadd"] = role_inout_group_friend["friendsnum_max"] - role_inout_group_friend["friendsnum_min"]

	df = role_inout_group_friend.groupby("uin").agg({"friendsnum_max":["sum", "mean", "max", std], "friendsadd":["sum", "mean", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	
	features_df = pd.merge(features_df, df, how='left', on='uin')

	# power features
	role_inout_group_power = role_inout_group.agg({"power":[min, max]}).reset_index()
	role_inout_group_power.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in role_inout_group_power.columns ]

	role_inout_group_power["poweradd"] = role_inout_group_power["power_max"] - role_inout_group_power["power_min"]

	df = role_inout_group_power.groupby("uin").agg({"power_max":["sum", "mean", "max", std], "poweradd":["sum", "mean", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	
	features_df = pd.merge(features_df, df, how='left', on='uin')
	
	# rolelevel features
	role_inout_group_rolelevel = role_inout_group.agg({"rolelevel":[min, max]}).reset_index()
	role_inout_group_rolelevel.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in role_inout_group_rolelevel.columns ]

	role_inout_group_rolelevel["roleleveladd"] = role_inout_group_rolelevel["rolelevel_max"] - role_inout_group_rolelevel["rolelevel_min"]

	df = role_inout_group_rolelevel.groupby("uin").agg({"rolelevel_max":["sum", "mean", "max", std], "roleleveladd":["sum", "mean", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	
	features_df = pd.merge(features_df, df, how='left', on='uin')
	
	# login-logout features
	df = role_inout_group.agg({"is_login": ["sum"], "onlinetime":["count"]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns ]

	df["login-logout"] = df["is_login_sum"] - df["onlinetime_count"]

	df = df.groupby("uin").agg({"is_login_sum":["sum", "mean", "max", std], "login-logout":["sum", "mean", "max", std]}).reset_index()
	df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
	
	# job features -- counts of each kind of jobs
	role_inout_group_job = role_inout_group.agg({"job": lambda x:x.iloc[0]}).reset_index()
	df = role_inout_group_job.join(pd.get_dummies(pd.get_dummies(role_inout_group_job.job, prefix="job")))
	df.drop(columns='job', inplace=True)

	df = df.groupby("uin").sum().reset_index().drop(columns=["roleid"])
	features_df = pd.merge(features_df, df, how='left', on='uin')

	return features_df

def organize_data(data_dir):

	labels_b = pd.read_csv(os.path.join(data_dir, "label_black/20200301.txt"), header=None, names=['uin'])
	labels_w = pd.read_csv(os.path.join(data_dir, "label_white/20200301.txt"), header=None, names=['uin'])

	labels_b["target"] = 1
	labels_w["target"] = 0
	labels = pd.concat([labels_b, labels_w], axis=0).reset_index(drop=True)

	print("读取login与logout数据ing...")
	role_login_tr, role_logout_tr = load_loginout(labels=labels.uin.values, data_dir=data_dir)
	role_login_te, role_logout_te = load_loginout(mode='test', data_dir=data_dir)

	print("role_login_tr:{}, role_logout_tr:{}".format(role_login_tr.shape, role_logout_tr.shape))
	print("role_login_te:{}, role_logout_te:{}".format(role_login_te.shape, role_logout_te.shape))

	role_login_tr = dropuna(role_login_tr)
	role_logout_tr = dropuna(role_logout_tr)
	role_login_te = dropuna(role_login_te)
	role_logout_te = dropuna(role_logout_te)
	print("role_login_tr:{}, role_logout_tr:{}".format(role_login_tr.shape, role_logout_tr.shape))
	print("role_login_te:{}, role_logout_te:{}".format(role_login_te.shape, role_logout_te.shape))

	printuninum([role_login_tr, role_logout_tr])
	printuninum([role_login_te, role_logout_te])

	role_inout_tr = concat_df(role_login_tr, role_logout_tr)
	role_inout_te = concat_df(role_login_te, role_logout_te)
	
	return role_login_tr, role_logout_tr, role_login_te, role_logout_te, role_inout_tr, role_inout_te, labels

def create_chat_df(data_dir=None, labels=None):
	print("读取create与chat数据ing...")
	role_create_tr, role_chat_tr = load_create_chat(labels=labels.uin.values, data_dir=data_dir)
	role_create_te, role_chat_te = load_create_chat(mode='test', data_dir=data_dir)

	print("role_create_tr:{}, role_chat_tr:{}".format(role_create_tr.shape, role_chat_tr.shape))
	print("role_create_te:{}, role_chat_te:{}".format(role_create_te.shape, role_chat_te.shape))

	role_create_tr = dropuna(role_create_tr)
	role_create_te = dropuna(role_create_te)
	role_create_tr["create_hour"] = role_create_tr.dteventtime.apply(lambda x: x.hour)
	role_create_te["create_hour"] = role_create_te.dteventtime.apply(lambda x: x.hour)
	print("role_create_tr:{}, role_create_te:{}".format(role_create_tr.shape, role_create_te.shape))
	
	return role_create_tr, role_chat_tr, role_create_te, role_chat_te

def money_item_df(data_dir=None, labels=None):
	print("读取moneyflow与itemflow数据ing...")
	role_money_tr, role_item_tr = load_money_item(labels=labels.uin.values, data_dir=data_dir)
	role_money_te, role_item_te = load_money_item(mode='test', data_dir=data_dir)

	print("role_money_tr:{}, role_item_tr:{}".format(role_money_tr.shape, role_item_tr.shape))
	print("role_money_te:{}, role_item_te:{}".format(role_money_te.shape, role_item_te.shape))
	
	return role_money_tr, role_item_tr, role_money_te, role_item_te
	
def get_money_features(money_or_item, name = "money", mode="train"):
	print("get_money_features ing...")
	feature_df = money_or_item[["uin"]].drop_duplicates().reset_index(drop=True)
	
	if "hour" not in money_or_item.columns:
		money_or_item["hour"] = money_or_item.dteventtime.apply(lambda x: x.hour).astype("int16")
		money_or_item["quarter"] = money_or_item["hour"]//6
		money_or_item.quarter = money_or_item.quarter.map({0:"0", 1:"1", 2:"2", 3:"3"})
	
	if isinstance(money_or_item.AddOrReduce.dtype, str):
		role_money_tr.AddOrReduce.map({"in":0, "out":1}).astype("int8")
	
	# hour level -- flowin flowout stats features
	df = money_or_item.groupby(['uin']).agg({"hour": unique_len}).reset_index()
	df.columns = ['uin', name+"_hour_nunqiue"]
	feature_df = feature_df.merge(df, how="left", on="uin")
	
	if name == "money":
		levels = ['roleid', 'hour', 'quarter', 'iMoneyType', 'Reason']
	else:
		levels = ['roleid', 'hour', 'quarter', 'itemid', 'Reason']
	
	# stats features of flowin or flowout counts 
	for _level in levels:
		
		print("Group by uins and ", _level)
		base_name = "{}_counts_{}".format(name, _level)

		df = money_or_item.groupby(['uin', _level]).agg({"rolelevel": "count", "AddOrReduce":"sum"}).reset_index()
		_d = {"rolelevel": base_name+"_sum", "AddOrReduce": base_name+"_out"}
		df.columns = [_d.get(i, i)  for i in df.columns.tolist()]
		if df.isna().sum().sum():
			print(df.isna().sum())
			df.fillna(0, inplace=True)

		df[base_name+"_in"] = df[base_name+"_sum"] - df[base_name+"_out"]
		df[base_name+"_diff"] = df[base_name+"_out"] - df[base_name+"_in"]

		_feats = ["sum", "mean", std] if _level == "roleid" else ["mean", std]
		print(_feats)

		df = df.groupby("uin").agg({base_name+"_in": _feats, base_name+"_out": _feats, base_name+"_diff": _feats, 
									base_name+"_sum": _feats}).reset_index()
		df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
		if df.isna().sum().sum():
			print(df.isna().sum())
			df.fillna(0, inplace=True)
			
		feature_df = feature_df.merge(df, how="left", on="uin")
			
	# stats features of flowin flowout counts 
	if name == "money":
		
		money_or_item.AddOrReduce = money_or_item.AddOrReduce.map({1:"out", 0:"in"})
		
		for _level in levels: # hour level -- flowin flowout stats features
			print("Group by uins and", _level)
			base_name = "iMoney_{}".format(_level)

			df = money_or_item.groupby(['uin', _level, "AddOrReduce"]).agg({"iMoney": ["sum", "max", "mean", "median", std]})
			df = df.unstack(fill_value=0).reset_index()
			df.columns = ["_".join(i).replace("iMoney", base_name) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]

			if df.isna().sum().sum():
				print(df.isna().sum())
				df.fillna(0, inplace=True)

			base_name += "_sum"
			df[base_name+"_sum"] = df[base_name+"_in"] + df[base_name+"_out"]
			df[base_name+"_diff"] = df[base_name+"_out"] - df[base_name+"_in"]
			
			_feats = ["sum", "mean", std] if _level == "roleid" else ["mean", std]
			print(_feats)
			
			if _level == "quarter":
				df_quarter = df.set_index(["uin", "quarter"]).unstack(fill_value=0).reset_index()
				df_quarter.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df_quarter.columns]

			df = df.groupby("uin").agg({base_name+"_in": _feats, base_name+"_out": _feats, base_name+"_diff": _feats, 
										base_name+"_sum": _feats}).reset_index()
			df.columns = ["_".join(i) if isinstance(i, tuple) and (i[1]!="") else i[0] for i in df.columns]
			if df.isna().sum().sum():
				print(df.isna().sum())
				df.fillna(0, inplace=True)

			if _level == "quarter":
				df = pd.merge(df, df_quarter, how="left", on="uin")
			
			feature_df = feature_df.merge(df, how="left", on="uin")
	
	feature_df.to_csv("./dataset/{}_money.csv".format(mode), index=False)
	# return feature_df

def merge_create_chat(df, role_create, role_chat):
	df = df.merge(role_create[['uin', 'create_hour']], how="left", on=['uin'])
	df.create_hour.fillna(-99, inplace=True)

	df['is_createtoday'] = 0 
	df.loc[df.create_hour>=0, 'is_createtoday']=1

	df = df.merge(role_chat, how="left", on=['uin'])
	df.chat_cnt.fillna(0, inplace=True)
	return df

def create_fe_csv(role_login, role_inout, mode='train', labels=None, role_create=None, role_chat=None):
	print("get_basic_features ing...")
	data_login = login_fe(role_login)
	if labels is not None:
		data_login = pd.merge(labels, data_login, how="left", on="uin")
	if data_login.isna().sum().sum():
		print(mode)
		print(data_login.isna().sum())
		data_login.fillna(0, inplace=True)
	data_inout = inout_fe(role_inout)
	data_inout.describe()
	final_data = pd.merge(data_inout, data_login, how="left", on=["uin"])
	if final_data.isna().sum().sum():
		print(mode)
		print(final_data.isna().sum())
		# final_data.fillna(0, inplace=True)
		
	final_data.login_len.fillna(0, inplace=True)
	final_data.login_dawn_hour_num.fillna(0, inplace=True)

	if not os.path.exists("./dataset"):
		os.makedirs("./dataset")
	
	final_data = pd.read_csv("./dataset/test_0.csv")
	role_create = pd.read_csv("./dataset/role_create.csv")
	role_chat = pd.read_csv("./dataset/role_chat.csv")
	print("role_create_tr:{}, role_create_te:{}".format(role_create.shape, role_chat.shape))
	final_data = merge_create_chat(final_data, role_create, role_chat)
	print("final_data:{}, role_create_te:{}".format(final_data.shape, role_chat.shape))
	final_data.to_csv("./dataset/{}_1.csv".format(mode), index=False)

def make(data_dir="/data1/studio_round1/"):
	
	role_login_tr, role_logout_tr, role_login_te, role_logout_te, role_inout_tr, role_inout_te, labels = organize_data(data_dir)
	role_create_tr, role_chat_tr, role_create_te, role_chat_te = create_chat_df(data_dir, labels)
	role_money_tr, role_item_tr, role_money_te, role_item_te = money_item_df(data_dir, labels)
	
	role_create_te.to_csv("./dataset/role_create.csv", index=False)
	role_chat_te.to_csv("./dataset/role_chat.csv", index=False)
	
	role_login_te["target"] = 0
	create_fe_csv(role_login_tr, role_inout_tr, 'train', labels, role_create_tr, role_chat_tr)
	create_fe_csv(role_login_te, role_inout_te, 'test', None, role_create_te, role_chat_te)
	
	get_money_features(role_money_tr, mode='train')
	get_money_features(role_money_te, mode='test')