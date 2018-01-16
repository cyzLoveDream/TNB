from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
#Validation function
n_folds = 10

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
y_train = trains["血糖"]
predictors = [f for f in list(trains.columns) if f not in ["血糖","blood_sugar","id","blood_sugar_log","体检日期"]]
def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男': 1,'女': 0})
    # data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    data.drop("体检日期",axis = 1,inplace= True)
    # data.fillna(data.median(axis=0))
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat,test_feat
params = {
    'learning_rate': 0.015,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'min_data_in_leaf': 25,
    'bagging_fraction':0.8,
    "feature_fraction":0.7,
    'min_sum_hessian_in_leaf': 1,
    'verbose': -1,
    "max_depth":9,
    "max_bin":150,
    "lambda_l2":0.3,
    "min_child_samples":15,
    "num_leaves":10
}
param_test = {
    'learning_rate': [0.001,0.005,0.01,0.05,0.1,0.5],
    # 'gamma':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
}
train, test = make_feat(trains, tests)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=10,boosting_type="gbdt",
                              learning_rate=0.015, n_estimators=3000,
                              max_bin = 150, bagging_fraction = 0.8,
                              feature_fraction = 0.7,
                              min_data_in_leaf =25, min_sum_hessian_in_leaf = 1,max_depth=9,
                              min_child_samples=15,reg_lambda=0.3,early_stopping_rounds=50)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.01, max_depth=7,
                             min_child_weight=1.7817, n_estimators=600,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.65, silent=1,
                             seed =1024, nthread = -1)

gsearch = GridSearchCV(model_xgb , param_grid = param_test, scoring='neg_mean_squared_error', cv=5)
def print_best_score(gsearch,param_test):

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
print("begin gridcv...")
gsearch.fit(train[predictors], y_train)
print_best_score(gsearch,param_test)