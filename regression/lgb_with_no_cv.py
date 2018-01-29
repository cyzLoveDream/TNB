import time
import datetime
import numpy as np
from dateutil.parser import parse
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

trains = pd.read_csv('../raw_data/d_train.csv',encoding="gbk")
tests = pd.read_csv("../raw_data/d_test_B_20180128.csv",encoding="gbk")
# fea_train = pd.read_csv("../raw_data/fea_train.csv")
# fea_test = pd.read_csv("../raw_data/fea_test.csv")
# trains.drop(trains[trains["年龄"] < 20].index,inplace=True)
trains.drop(trains[trains["年龄"] >= 86].index,inplace=True)
# trains.drop(trains[trains["血糖"] > 15].index,inplace=True)
# trains = pd.read_csv("../clean_data/base_train.csv")
# vals = pd.read_csv('../clean_data/model_fill_train.csv')
# tests = pd.read_csv('../clean_data/test.csv')
# trains = pd.concat([trains, vals], axis=0, ignore_index=True)
fea_train = pd.read_csv("../raw_data/fea_train.csv")
fea_test = pd.read_csv("../raw_data/fea_test_B.csv")
fea_train1 = pd.read_csv("../raw_data/fea_train_1.csv")
fea_test1 = pd.read_csv("../raw_data/fea_test_B_1.csv")
fea_train2 = pd.read_csv("../raw_data/fea_train_2.csv")
fea_test2 = pd.read_csv("../raw_data/fea_test_B_2.csv")
# fea_train3 = pd.read_csv("../raw_data/fea_train_3.csv")
# fea_test3 = pd.read_csv("../raw_data/fea_test_3.csv")
fea_mul_train = pd.read_csv("../raw_data/fea_mul_train.csv")
fea_mul_test = pd.read_csv("../raw_data/fea_mul_test.csv")
fea_df_test = pd.read_csv("../raw_data/fea_df_test.csv")
fea_df_train = pd.read_csv("../raw_data/fea_df_train.csv")
trains = pd.merge(trains, fea_train, how="left",on="id")
trains = pd.merge(trains, fea_train1, how="left",on="id")
trains = pd.merge(trains, fea_train2, how="left",on="id")
# trains = pd.merge(trains, fea_train3, how="left",on="id")
# trains = pd.merge(trains, fea_df_train, how="left",on="id")
# trains = pd.merge(trains, fea_mul_train, how="left",on="id")
tests = pd.merge(tests, fea_test, how="left",on="id")
tests = pd.merge(tests, fea_test1, how="left",on="id")
tests = pd.merge(tests, fea_test2, how="left",on="id")
# tests = pd.merge(tests, fea_test3, how="left",on="id")
# tests = pd.merge(tests, fea_df_test, how="left",on="id")
# tests = pd.merge(tests, fea_mul_test, how="left",on="id")
def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    # train = train[train['年龄']>=16]
    # train = train[train['血糖']<=18]
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男': 1, '女': 0,'??':1})
    # data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    # data.drop("体检日期",axis = 1,inplace= True)
    # data['体检日期'] = pd.to_datetime(data['体检日期'])
    # data['weekday'] = data['体检日期'].apply(lambda r: r.weekday())
    # data['weekendFlag']=(data['weekday']>5)+0
    # data['weekday'] = data['weekday'].apply(lambda r: 'd'+str((r+1)))
# data.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','体检日期'], axis=1, inplace=True)
    # data.fillna(data.mean(axis=0),inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat,test_feat
train, test = make_feat(trains, tests)
# '*r-谷氨酰基转换酶', '*丙氨酸氨基转换酶', '*天门冬氨酸氨基转换酶', '*总蛋白', '*球蛋白', '*碱性磷酸酶',
#  'ATSm', 'ATSs', 'feature_0_feature_1', 'feature_0_is_normal',
#  'feature_10_feature_11', 'feature_10_gender_is_normal',
#  'feature_10_is_normal', 'feature_11_is_normal', 'feature_12_feature_13',
#  'feature_12_is_normal', 'feature_13_is_normal',
#  'feature_14_gender_is_normal', 'feature_14_is_normal',
#  'feature_1_feature_0', 'feature_1_feature_0_is_normal',
#  'feature_1_is_normal', 'feature_20_feature_32', 'feature_20_feature_33',
#  'feature_20_feature_34', 'feature_20_feature_35',
#  'feature_20_feature_36', 'feature_20_is_normal', 'feature_21_is_normal',
#  'feature_22_is_normal', 'feature_23_is_normal', 'feature_24_is_normal',
#  'feature_25_is_normal', 'feature_26_is_normal', 'feature_27_is_normal',
#  'feature_28_feature_31', 'feature_28_is_normal', 'feature_29_is_normal',
#  'feature_2_is_normal', 'feature_30_is_normal', 'feature_31_is_normal',
#  'feature_32_feature_33', 'feature_32_is_normal', 'feature_33_is_normal',
#  'feature_34_is_normal', 'feature_35_is_normal', 'feature_36_is_normal',
#  'feature_3_is_normal', 'feature_4_feature_5', 'feature_4_feature_6',
#  'feature_4_is_normal', 'feature_4_less_60', 'feature_5_is_normal',
#  'feature_5_less_25', 'feature_6_is_normal', 'feature_8_feature_9',
#  'feature_8_is_normal', 'feature_9_is_normal', 'id', '中性粒细胞%', '乙肝e抗体',
#  '乙肝e抗原', '乙肝核心抗体', '乙肝表面抗体', '乙肝表面抗原', '低密度脂蛋白胆固醇', '单核细胞%', '嗜碱细胞%',
#  '嗜酸细胞%', '尿素', '尿酸', '年龄', '性别', '总胆固醇', '淋巴细胞%', '甘油三酯', '白球比例',
#  '白细胞计数', '白蛋白', '红细胞体积分布宽度', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白浓度',
#  '红细胞平均血红蛋白量', '红细胞计数', '肌酐', '血小板体积分布宽度', '血小板平均体积', '血小板比积', '血小板计数',
#  '血糖', '血红蛋白', '高密度脂蛋白胆固醇'
print(train.shape, test.shape)
def evalerror(pred,df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred) * 0.5
    return ('mse',score,False)
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
# 'feature_26_is_normal','红细胞平均血红蛋白量','feature_20_feature_33','feature_28_is_normal','feature_10_gender_is_normal',
# '淋巴细胞%','白细胞计数','feature_1_is_normal','血小板计数'
# print(list(train.columns))
# for i in list(train.columns):
no_use = ["血糖","blood_sugar","id","blood_sugar_log",'体检日期','feature_5_less_25','feature_4_less_60','性别']
# if i not in no_use:
#     no_use.append(i)
# else:
#     continue
predictors = [f for f in list(train.columns) if f not in no_use]
X_train, X_test, y_train,y_test = train_test_split(train[predictors], train["血糖"],test_size=0.1,random_state=42)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test)
lgb_test = lgb.Dataset(test)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=lgb_eval,
                verbose_eval=False,
                feval=evalerror,
                early_stopping_rounds=100)
test_pred = gbm.predict(test[predictors],num_iteration=gbm.best_iteration)
# submission = pd.DataFrame({'pred': test_pred})
# feat_imp = pd.DataFrame()
# feat_i = pd.DataFrame(pd.Series(gbm.feature_importance(),index=predictors).sort_values(ascending=False))
# feat_imp = pd.concat([feat_imp, feat_i],axis=1)
# feat_imp.to_csv("./feature_imp.csv",header=False)
test["pred"] = test_pred
val_pred = gbm.predict(X_test[predictors], num_iteration=gbm.best_iteration)
print("线下误差：{} 最大值：{} 314的值{} 不加特征： {}".format((mean_squared_error(y_test,  val_pred) * 0.5), test["pred"].max(),test[test.id.values == 6054]["pred"].values ,1))
print(test["pred"].describe(), train["血糖"].describe(),pd.DataFrame(val_pred).describe(),y_test.describe())
# test[['id',"pred"]].to_csv(r'./sub_B.csv',index=False)
test[["pred"]].to_csv(r'../submission/lgb.csv',index=False,header=False)