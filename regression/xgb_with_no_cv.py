import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
xgb.cv

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
    data.drop("体检日期",axis = 1,inplace= True)
    # data.fillna(value = -1, inplace = True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat,test_feat
def evalerror(pred,df):
    label = df.get_label()
    score = mean_squared_error(label,pred) * 0.5
    return ('mse',score)

train, test = make_feat(trains, tests)
predictors = [f for f in list(train.columns) if f not in ["血糖","blood_sugar","id","blood_sugar_log","体检日期"]]
print('开始训练...')
params = {
    "objective":"reg:linear",
    "eta":0.015,
    "min_child_weight":1,
    "subsample":0.65,
    "colsample_bytree":0.8,
    "lambda":5,
    "seed":1024,
    "silent":1,
    "verbose":0,
    "max_depth":7,
    "alpha":0.01,
    "gamma":0.6
}
X_train, X_test, y_train,y_test = train_test_split(train[predictors], train["血糖"],test_size=0.1,random_state=42)
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_eval = xgb.DMatrix(X_test, label=y_test)
xgb_test = xgb.DMatrix(test[predictors])
watchlist = [(xgb_train,"train"),(xgb_eval,'val')]
xgb_model = xgb.train(params, xgb_train,num_boost_round=3000,
                      early_stopping_rounds=50,
                      evals=watchlist,
                      feval=evalerror,verbose_eval=100)
test_pred = xgb_model.predict(xgb_test,ntree_limit=xgb_model.best_ntree_limit)
# submission = pd.DataFrame({'pred': test_pred})
test["pred"] = test_pred
val_pred = xgb_model.predict(xgb_eval,ntree_limit=xgb_model.best_ntree_limit)
print("线下误差：", mean_squared_error(y_test,  val_pred) * 0.5)
print(test["pred"].describe(), train["血糖"].describe(),pd.DataFrame(val_pred).describe() ,y_test.describe())
test[["id","pred"]].to_csv(r'../regression/sub_xgb_16_2_h.csv',index=False)