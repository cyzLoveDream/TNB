import pandas as pd
import numpy as np
import xgboost as xgb
import time
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

print("input data...")
trains = pd.read_csv('../raw_data/d_train.csv',encoding="gbk")
tests = pd.read_csv("../raw_data/d_test_A.csv",encoding="gbk")
trains.drop(trains[trains["年龄"] >= 84].index,inplace=True)
fea_train = pd.read_csv("../raw_data/fea_train.csv")
fea_test = pd.read_csv("../raw_data/fea_test.csv")
fea_train1 = pd.read_csv("../raw_data/fea_train_1.csv")
fea_test1 = pd.read_csv("../raw_data/fea_test_1.csv")
trains = pd.merge(trains, fea_train, how="left",on="id")
trains = pd.merge(trains, fea_train1, how="left",on="id")
tests = pd.merge(tests, fea_test, how="left",on="id")
tests = pd.merge(tests, fea_test1, how="left",on="id")
trains["血糖"] = trains["血糖"].apply(lambda x: 1 if x < 4.5 else 0)
trains["血糖"] = trains["血糖"].astype(np.int32)
print("finish input...")
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
train, test = make_feat(trains, tests)
print("train's shape is: ",train.shape)
print("test's shape is: ",test.shape)
predictors = [f for f in list(train.columns) if f not in ["血糖","blood_sugar","id","blood_sugar_log","体检日期"]]
# X_train, X_test, y_train,y_test = train_test_split(train[predictors], train["血糖"],test_size=0.1,random_state=42)
xgb_train = xgb.DMatrix(train[predictors], label=train["血糖"])
# xgb_eval = xgb.DMatrix(X_test, label=y_test)
xgb_test = xgb.DMatrix(test[predictors])
params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'scale_pos_weight':float(len(train["血糖"]) - sum(train["血糖"])) / sum(train["血糖"]),
    'eval_metric': 'auc',
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
watchlist  = [(xgb_train,'train')]
#通过cv找最佳的nround
print("find best num round by cv..")
cv_log = xgb.cv(params,xgb_train,num_boost_round=25000,nfold=5,metrics='auc',early_stopping_rounds=50,seed=1024)
bst_auc= cv_log['test-auc-mean'].max()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
bst_nb = cv_log.nb.to_dict()[bst_auc]
print("best test auc is {0}, best num round is {1}".format(bst_auc, bst_nb))
#train
watchlist  = [(xgb_train,'train')]
print("train...")
model = xgb.train(params,xgb_train,num_boost_round=bst_nb+50,evals=watchlist)
test_pred = model.predict(xgb_test)
test_result = test
test_result["lt45_prob"] = test_pred
# test_result[["id","lt45_prob"]].to_csv("./result/lt45.csv",index=False)
#