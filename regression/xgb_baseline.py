import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
trains = pd.read_csv('../raw_data/d_train.csv',encoding="gbk")
tests = pd.read_csv("../raw_data/d_test_A.csv",encoding="gbk")
# trains.drop(trains[trains["年龄"] <= 16].index,inplace=True)
trains.drop(trains[trains["年龄"] >= 84].index,inplace=True)
# trains.drop(trains[trains["血糖"] > 15].index,inplace=True)
# trains = pd.read_csv("./clean_data/base_train.csv")
# vals = pd.read_csv('./clean_data/model_fill_train.csv')
# tests = pd.read_csv('./clean_data/test.csv')
# trains = pd.concat([trains, vals], axis=0, ignore_index=True)
fea_train = pd.read_csv("../raw_data/fea_train.csv")
fea_test = pd.read_csv("../raw_data/fea_test.csv")
fea_train1 = pd.read_csv("../raw_data/fea_train_1.csv")
fea_test1 = pd.read_csv("../raw_data/fea_test_1.csv")
trains = pd.merge(trains, fea_train, how="left",on="id")
trains = pd.merge(trains, fea_train1, how="left",on="id")
tests = pd.merge(tests, fea_test, how="left",on="id")
tests = pd.merge(tests, fea_test1, how="left",on="id")

def make_feat(train,test):
	train_id = train.id.values.copy()
	test_id = test.id.values.copy()
	data = pd.concat([train,test])
	data['性别'] = data['性别'].map({'男': 1,'女': 0})
	# data["性别"] = data['性别'].astype(int)
	# data['date'] = (pd.to_datetime(data['date']) - parse('2017-10-09')).dt.days
	# data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
	# data.drop("体检日期",axis = 1,inplace= True)
	# data.fillna(value = -1, inplace = True)
	train_feat = data[data.id.isin(train_id)]
	test_feat = data[data.id.isin(test_id)]
	return train_feat,test_feat
train, test = make_feat(trains, tests)
# no_use = ["feature_6_is_normal","feature_4_is_normal","feature_36_is_normal","feature_35_is_normal","feature_34_is_normal","feature_33_is_normal",
#           "feature_32_is_normal","feature_31_is_normal","feature_30_is_normal","feature_2_is_normal","feature_29_is_normal","feature_28_is_normal",
#           "feature_27_is_normal","feature_25_is_normal","feature_24_is_normal","feature_20_is_normal","feature_1_is_normal","feature_12_is_normal",
#           ]
# use = [x for x in list(train.columns) if x not in no_use]
# train = train[use]
# test = test[use]
print(train.shape)
print(test.shape)
predictors = [f for f in list(train.columns) if f not in ["血糖","blood_sugar","id","blood_sugar_log","体检日期"]]

print('开始训练...')
params = {
	"objective":"reg:linear",
	"eta":0.015,
	"min_child_weight":1,
	"subsample":0.8,
	"colsample_bytree":0.8,
	"lambda":0.1,
	"seed":1024,
	"silent":1,
	"verbose":0,
	"max_depth":6,
	"alpha":0,
	"gamma":0.3
	}
def evalerror(pred,df):
	label = df.get_label()
	score = mean_squared_error(label,pred) * 0.5
	return ('mse',score)
print('开始CV 5折训练...')
for p in [256]:
	params["seed"] = p
	scores = []
	t0 = time.time()
	train_preds = np.zeros(train.shape[0])
	test_preds = np.zeros((test.shape[0],10))
	feat_imp = pd.DataFrame()
	kf = KFold(len(train),n_folds=10,shuffle=True,random_state=1024)
	xgb_test = xgb.DMatrix(test[predictors])
	# model_log = xgb.cv(params, xgb.DMatrix(train[predictors],label=train["血糖"]), num_boost_round=500,nfold=5,early_stopping_rounds=10,
	#        verbose_eval=100,seed=42)
	# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
	# plt.show()
	# print(model_log)
	for i,(train_index,test_index) in enumerate(kf):
		# print('第{}次训练...'.format(i))
		train_feat1 = train.iloc[train_index]
		train_feat2 = train.iloc[test_index]
		xgb_train1 = xgb.DMatrix(train_feat1[predictors],label=train_feat1["血糖"])
		xgb_train2 = xgb.DMatrix(train_feat2[predictors],label=train_feat2["血糖"])
		watchlist = [(xgb_train2,'val')]
		
		xgb_model = xgb.train(params, xgb_train1,num_boost_round=3000,
		                      early_stopping_rounds=50,
		                      evals=watchlist,
		                      feval=evalerror,verbose_eval=False)
		train_preds[test_index] += xgb_model.predict(xgb_train2,ntree_limit = xgb_model.best_ntree_limit)
		test_preds[:,i] = xgb_model.predict(xgb_test,ntree_limit = xgb_model.best_ntree_limit)
	print('线下得分:{0}, seed: {1}'.format((mean_squared_error(train['血糖'],train_preds) * 0.5), params.get("seed")))
	print('CV训练用时{}秒'.format(time.time() - t0))
	submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
	print(submission.describe(), train["血糖"].describe())
	submission.to_csv(r'../submission/sub_xgb_17_2_h.csv',header=False,index=False)
