import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

trains = pd.read_csv('../raw_data/d_train.csv',encoding="gbk")
tests = pd.read_csv("../raw_data/d_test_A.csv",encoding="gbk")
# fea_train = pd.read_csv("../raw_data/fea_train.csv")
# fea_test = pd.read_csv("../raw_data/fea_test.csv")
# trains.drop(trains[trains["年龄"] < 20].index,inplace=True)
trains.drop(trains[trains["年龄"] >= 84].index,inplace=True)
# trains.drop(trains[trains["血糖"] > 15].index,inplace=True)
# trains = pd.read_csv("../clean_data/base_train.csv")
# vals = pd.read_csv('../clean_data/model_fill_train.csv')
# tests = pd.read_csv('../clean_data/test.csv')
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
    # data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    data.drop("体检日期",axis = 1,inplace= True)
    # data.fillna(data.mean(axis=0),inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat,test_feat
train, test = make_feat(trains, tests)
predictors = [f for f in list(train.columns) if f not in ["血糖","blood_sugar","id","blood_sugar_log","体检日期"]]
def evalerror(pred,df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred) * 0.5
    return ('mse',score,False)

X_train, X_test, y_train,y_test = train_test_split(train[predictors], train["血糖"],test_size=0.1,random_state=42)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test)
lgb_test = lgb.Dataset(test)
print('开始训练...')
params = {
    'learning_rate': 0.015,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 12,
    'max_depth':9,
    'max_bin':130,
    'feature_fraction': 0.9,
    'reg_lambda':50,
    'min_data': 25,
    'min_child_weight':0.001,
    'verbose': -1,
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=lgb_eval,
                verbose_eval=100,
                feval=evalerror,
                early_stopping_rounds=50)
test_pred = gbm.predict(test[predictors],num_iteration=gbm.best_iteration)
# submission = pd.DataFrame({'pred': test_pred})
test["pred"] = test_pred
val_pred = gbm.predict(X_test[predictors], num_iteration=gbm.best_iteration)
print("线下误差：", mean_squared_error(y_test,  val_pred) * 0.5)
print(test["pred"].describe(), train["血糖"].describe(),pd.DataFrame(val_pred).describe() ,y_test.describe())
# submission.to_csv(r'../regression/sub_lgb_16_1_c.csv',header=False,index=False)