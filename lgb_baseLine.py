import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
trains = pd.read_csv('./raw_data/d_train.csv',encoding="gbk")
tests = pd.read_csv("./raw_data/d_test_A.csv",encoding="gbk")
# fea_train = pd.read_csv("./raw_data/fea_train.csv")
# fea_test = pd.read_csv("./raw_data/fea_test.csv")
# trains.drop(trains[trains["年龄"] < 20].index,inplace=True)
trains.drop(trains[trains["年龄"] >= 84].index,inplace=True)
# trains.drop(trains[trains["血糖"] > 15].index,inplace=True)
# trains = pd.read_csv("./clean_data/base_train.csv")
# vals = pd.read_csv('./clean_data/model_fill_train.csv')
# tests = pd.read_csv('./clean_data/test.csv')
# trains = pd.concat([trains, vals], axis=0, ignore_index=True)
# fea_train = pd.read_csv("./raw_data/fea_train.csv")
# fea_test = pd.read_csv("./raw_data/fea_test.csv")
# fea_train1 = pd.read_csv("./raw_data/fea_train_1.csv")
# fea_test1 = pd.read_csv("./raw_data/fea_test_1.csv")
# trains = pd.merge(trains, fea_train, how="left",on="id")
# trains = pd.merge(trains, fea_train1, how="left",on="id")
# tests = pd.merge(tests, fea_test, how="left",on="id")
# tests = pd.merge(tests, fea_test1, how="left",on="id")

def make_feat(train,test):
	train_id = train.id.values.copy()
	test_id = test.id.values.copy()
	data = pd.concat([train,test])
	data['性别'] = data['性别'].map({'男': 1,'女': 0})
	# data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
	# data.drop("体检日期",axis = 1,inplace= True)
	data.fillna(data.mean(axis=0),inplace=True)
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
# gender_onehot_train = pd.get_dummies(train['gender'])
# gender_onehot_train.rename(columns={0:'gender_0',1:'gender_1'},inplace=True)
# train = pd.concat([train, gender_onehot_train],axis=1)
# train.drop("gender",axis=1,inplace = True)
# gender_onehot_test = pd.get_dummies(test['gender'])
# gender_onehot_test.rename(columns={0:'gender_0',1:'gender_1'},inplace=True)
# test = pd.concat([test, gender_onehot_test],axis=1)
# test.drop("gender",axis=1,inplace = True)
# # train_feat,test_feat = make_feat(train,test)
# predictors = [f for f in list(train.columns) if f not in ["blood_sugar"]]
# train["blood_sugar"] = trains["blood_sugar"]
print()
def evalerror(pred,df):
	label = df.get_label().values.copy()
	score = mean_squared_error(label,pred) * 0.5
	return ('mse',score,False)


print('开始训练...')
params = {
    'learning_rate': 0.005,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 10,
	'max_depth':9,
	'max_bin':130,
    'colsample_bytree': 0.8,
    'feature_fraction': 0.7,
    'min_data': 20,
    'min_hessian': 1,
    'verbose': -1,
}
print('开始CV 5折训练...')

for i in [20]:
	params["min_data"] = i
	scores = []
	t0 = time.time()
	train_preds = np.zeros(train.shape[0])
	test_preds = np.zeros((test.shape[0],10))
	# feat_imp = pd.DataFrame()
	kf = KFold(len(train),n_folds=10,shuffle=True,random_state=1024)
	for i,(train_index,test_index) in enumerate(kf):
		# print('第{}次训练...'.format(i))
		train_feat1 = train.iloc[train_index]
		train_feat2 = train.iloc[test_index]
		lgb_train1 = lgb.Dataset(train_feat1[predictors],train_feat1['血糖'])
		lgb_train2 = lgb.Dataset(train_feat2[predictors],train_feat2['血糖'])
		gbm = lgb.train(params,
		                lgb_train1,
		                num_boost_round=3000,
		                valid_sets=lgb_train2,
		                verbose_eval=False,
		                feval=evalerror,
		                early_stopping_rounds=50)
		feat_i = pd.DataFrame(pd.Series(gbm.feature_importance(),index=predictors).sort_values(ascending=False))
		# feat_imp = pd.concat([feat_imp, feat_i],axis=1)
		train_preds[test_index] += gbm.predict(train_feat2[predictors],num_iteration=gbm.best_iteration)
		test_preds[:,i] = gbm.predict(test[predictors],num_iteration=gbm.best_iteration)
	# print(feat_imp)
	# feat_imp.to_csv("./feature_imp.csv",header=False)
	print('线下得分:{0}, min_data: {1}'.format((mean_squared_error(train['血糖'],train_preds) * 0.5),params.get("min_data")))
	print('CV训练用时{}秒'.format(time.time() - t0))
	submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
	print(submission.describe(), train["血糖"].describe())
	# submission.to_csv(r'./submission/sub_lgb_15_1_c.csv',header=False,index=False)
