import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor

class b_model:
    # 这个地方可以定义全局变量
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
    no_use = ["血糖","blood_sugar","id","blood_sugar_log",'体检日期','feature_5_less_25','feature_4_less_60','性别']
    def __init__(self):
        # 在创建类的时候需要哪些参数
        self.model = LGBMRegressor(learning_rate=0.015,objective="regression",metric='mse',
                                   num_leaves = 12,max_depth=9, max_bin = 130,feature_fraction=0.9, reg_lambda=50,min_data = 25,min_child_weight=0.001,num_boost_round=3000,random_state=42)

    def __make_feature(self, train, test):
        # 构造特征
        if train.empty:
            test['性别'] = test['性别'].map({'男': 1, '女': 0,'??':1})
            return test
        if test.empty:
            train['性别'] = train['性别'].map({'男': 1, '女': 0,'??':1})
            return train
        else:
            train_id = train.id.values.copy()
            test_id = test.id.values.copy()
            data = pd.concat([train,test])
            data['性别'] = data['性别'].map({'男': 1, '女': 0,'??':1})
            train_feat = data[data.id.isin(train_id)]
            test_feat = data[data.id.isin(test_id)]
            return train_feat,test_feat

    def fit(self, X, y = None):
        X.drop(X[X["年龄"] >= 84].index,inplace=True)
        fea_train = pd.read_csv("./feature/fea_train.csv")
        fea_train1 = pd.read_csv("./feature/fea_train_1.csv")
        fea_train2 = pd.read_csv("./feature/fea_train_2.csv")
        X = pd.merge(X, fea_train, how="left",on="id")
        X = pd.merge(X, fea_train1, how="left",on="id")
        X = pd.merge(X, fea_train2, how="left",on="id")
        X = self.__make_feature(train = X, test=pd.DataFrame())
        if y == None:
            y = X["血糖"].values
        predictors = [f for f in list(X.columns) if f not in self.no_use]
        X_train, X_test, y_train,y_test = train_test_split(X[predictors], y,test_size=0.1,random_state=42)
        self.model.fit(X_train[predictors], y_train,eval_metric="mse",early_stopping_rounds=100,verbose=100,eval_set=(X_test[predictors], y_test))
        from sklearn.metrics import mean_squared_error
        print("线下误差：{}".format(0.5 * mean_squared_error(y_test,self.model.predict(X_test[predictors]))))
        return self

    def predict(self, X):
        # 对测试集进行预测,传入模型，和测试数据
        fea_test = pd.read_csv("./feature/fea_test.csv")
        fea_test1 = pd.read_csv("./feature/fea_test_1.csv")
        fea_test2 = pd.read_csv("./feature/fea_test_2.csv")
        X = pd.merge(X, fea_test, how="left",on="id")
        X = pd.merge(X, fea_test1, how="left",on="id")
        X = pd.merge(X, fea_test2, how="left",on="id")
        X = self.__make_feature(test=X, train = pd.DataFrame())
        predictors = [f for f in list(X.columns) if f not in self.no_use]
        test_pred = self.model.predict(X[predictors])
        print("最大值：{}".format(test_pred.max()))
        return test_pred

    def get_params(self):
        return self.params

biorad = b_model()
train = pd.read_csv("../raw_data/d_train.csv",encoding="gbk")
test = pd.read_csv("../raw_data/d_test_A.csv",encoding="gbk")
lgb = biorad.fit(train)
lgb.predict(test)


