import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import xgboost as xgb
xgb.XGBRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
trains = pd.read_csv('../clean_data/base_train.csv',encoding="gbk")
tests = pd.read_csv('../clean_data/test.csv',encoding='gbk')
vals = pd.read_csv('../clean_data/model_fill_train.csv',encoding='gbk')
labels = "blood_sugar_log"
features = [x for x in list(trains.columns) if x not in ['id','date','blood_sugar','blood_sugar_log']]
X_val, X_test, y_val, y_test = train_test_split(vals[features], vals[labels],test_size=0.1, random_state=42)
all_data["feature_9_is_normal"] = all_data["feature_9"].apply(lambda x: 0 if x < 2.9 else (2 if x > 6 else 1))

all_data["feature_10_is_normal"] = all_data["feature_10"].apply(lambda x: 0 if x < 1.16 else (2 if x > 1.55 else 1))
all_data["feature_10_gender_is_normal"] = all_data[["feature_10","gender"]].apply(lambda x:
                                       (0 if x[0] < 1.1 else 1) if x[1] == 1 else (2 if x[0] < 1.2 else 3),
                                       axis=1)
all_data["feature_11_is_normal"] = all_data["feature_11"].apply(lambda x: 0 if x < 2.84 else (2 if x > 3.12 else 1))
all_data["feature_10_feature_11"] = round(all_data["feature_11"] / all_data['feature_10'],3)
all_data["feature_12_is_normal"] = all_data["feature_12"].apply(lambda x: 0 if x < 1.7 else (2 if x > 8.3 else 1))
all_data["feature_13_is_normal"] = all_data[["feature_13","gender"]].apply(lambda x:
                                       (0 if x[0] < 53 else(2 if x[0] > 106 else 1)) if x[1] == 1 else (3 if x[0] < 44 else(5 if x[0] > 97 else 4)),
                                       axis=1)
all_data["feature_14_is_normal"] = all_data[["feature_14","gender"]].apply(lambda x:
                                       (0 if x[0] < 150 else(2 if x[0] > 416 else 1)) if x[1] == 1 else (3 if x[0] < 89 else(5 if x[0] > 357 else 4)),
                                       axis=1)
all_data["feature_14_gender_is_normal"] = all_data[["feature_14","gender"]].apply(lambda x:
                                       (0 if x[0] > 420 else 1) if x[1] == 1 else (2 if x[0] > 350 else 3),
                                       axis=1)
all_data["feature_20_is_normal"] = all_data["feature_20"].apply(lambda x: 0 if x < 4 else (2 if x > 10 else 1))
all_data["feature_21_is_normal"] = all_data[["feature_21","gender"]].apply(lambda x:
                                       (0 if x[0] < 4.0 else(2 if x[0] > 5.5 else 1)) if x[1] == 1 else (3 if x[0] < 3.5 else(5 if x[0] > 5.0 else 4)),
                                       axis=1)
all_data["feature_22_is_normal"] = all_data[["feature_22","gender"]].apply(lambda x:
                                       (0 if x[0] < 120 else(2 if x[0] > 160 else 1)) if x[1] == 1 else (3 if x[0] < 110 else(5 if x[0] > 150 else 4)),
                                       axis=1)
all_data["feature_23_is_normal"] = all_data[["feature_23","gender"]].apply(lambda x:
                                       (0 if x[0] < 0.4 else(2 if x[0] > 0.5 else 1)) if x[1] == 1 else (3 if x[0] < 0.37 else(5 if x[0] > 0.48 else 4)),
                                       axis=1)
all_data["feature_24_is_normal"] = all_data["feature_24"].apply(lambda x: 0 if x < 80 else (2 if x > 100 else 1))
all_data["feature_25_is_normal"] = all_data["feature_25"].apply(lambda x: 0 if x < 27 else (2 if x > 34 else 1))
all_data["feature_26_is_normal"] = all_data["feature_26"].apply(lambda x: 0 if x < 320 else (2 if x > 360 else 1))
all_data["feature_27_is_normal"] = all_data["feature_27"].apply(lambda x: 0 if x < 11.5 else (2 if x > 14.5 else 1))
all_data["feature_28_is_normal"] = all_data["feature_28"].apply(lambda x: 0 if x < 100 else (2 if x > 300 else 1))
all_data["feature_29_is_normal"] = all_data["feature_29"].apply(lambda x: 0 if x < 9 else (2 if x > 13 else 1))
all_data["feature_30_is_normal"] = all_data["feature_30"].apply(lambda x: 0 if x < 9 else (2 if x > 17 else 1))
all_data["feature_31_is_normal"] = all_data["feature_31"].apply(lambda x: 0 if x < 0.13 else (2 if x > 0.43 else 1))
all_data["feature_32_is_normal"] = all_data["feature_32"].apply(lambda x: 0 if x < 50 else (2 if x > 70 else 1))
all_data["feature_20_feature_32"] = round(all_data["feature_20"] * all_data['feature_32'],3)
all_data["feature_33_is_normal"] = all_data["feature_33"].apply(lambda x: 0 if x < 20 else (2 if x > 40 else 1))
all_data["feature_20_feature_33"] = round(all_data["feature_20"] * all_data['feature_33'],3)
all_data["feature_34_is_normal"] = all_data["feature_34"].apply(lambda x: 0 if x < 3 else (2 if x > 8 else 1))
all_data["feature_20_feature_34"] = round(all_data["feature_20"] * all_data['feature_34'],3)
all_data["feature_35_is_normal"] = all_data["feature_35"].apply(lambda x: 0 if x < 0.5 else (2 if x > 5 else 1))
all_data["feature_20_feature_35"] = round(all_data["feature_20"] * all_data['feature_35'],3)
all_data["feature_36_is_normal"] = all_data["feature_36"].apply(lambda x: 0 if x < 0 else (2 if x > 1 else 1))
all_data["feature_20_feature_36"] = round(all_data["feature_20"] * all_data['feature_36'],3)
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=550,
                              max_bin = 25, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =7, min_sum_hessian_in_leaf = 12)
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

[ 'feature_0_is_normal',
       'feature_1_is_normal', 'feature_1_feature_0',
       'feature_1_feature_0_is_normal', 'feature_2_is_normal',
       'feature_3_is_normal', 'feature_4_is_normal', 'feature_5_is_normal',
       'feature_6_is_normal', 'feature_8_is_normal', 'feature_9_is_normal',
       'feature_10_is_normal', 'feature_10_gender_is_normal',
       'feature_11_is_normal', 'feature_10_feature_11', 'feature_12_is_normal',
       'feature_13_is_normal', 'feature_14_is_normal',
       'feature_14_gender_is_normal', 'feature_20_is_normal',
       'feature_21_is_normal', 'feature_22_is_normal', 'feature_23_is_normal',
       'feature_24_is_normal', 'feature_25_is_normal', 'feature_26_is_normal',
       'feature_27_is_normal', 'feature_28_is_normal', 'feature_29_is_normal',
       'feature_30_is_normal', 'feature_31_is_normal', 'feature_32_is_normal',
       'feature_20_feature_32', 'feature_33_is_normal',
       'feature_20_feature_33', 'feature_34_is_normal',
       'feature_20_feature_34', 'feature_35_is_normal',
       'feature_20_feature_35', 'feature_36_is_normal',
       'feature_20_feature_36', 'gender_0', 'gender_1']

