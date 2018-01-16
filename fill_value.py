from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def fill_test(file_name,base_name):
	base_train = pd.read_csv(base_name,encoding='gbk')
	df_test_nan = pd.read_csv(file_name,encoding='gbk')
	cols = list(df_test_nan.columns)
	with_nan_cols = [k for k,v in zip(list(df_test_nan.columns),list(df_test_nan.isnull().any())) if
	                 v == True and k not in ['id',"date","blood_sugar","blood_sugar_log"]]
	print("have nan's columns: ", with_nan_cols)
	for c in cols:
		rf_pre_data = df_test_nan[df_test_nan[c].isnull().values == True]
		if rf_pre_data.shape[0] == 0:
			continue
		no_index = [x for x in list(df_test_nan.index) if x not in list(rf_pre_data.index)]
		rf_no_pre_data = df_test_nan.ix[no_index,:]
		all_rf_data = pd.concat([rf_no_pre_data,rf_pre_data],axis=0)
		no_nan_cols = [k for k,v in zip(list(rf_pre_data.columns),list(rf_pre_data.isnull().any())) if
		               v == False and k not in ['id',"date","blood_sugar","blood_sugar_log"]]
		mean = round((df_test_nan[c].mean() + base_train[c].mean()) / 2 ,3)
		trains = np.array(base_train[no_nan_cols])
		labels = np.array(base_train[c])
		tests = np.array(rf_pre_data[no_nan_cols])
		# rf = RandomForestRegressor(n_estimators = 20,min_samples_leaf = 20,random_state = 1024)
		lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=42))
		lasso.fit(trains,labels)
		rf_pre_data[c] = [round((2 * x * mean) / (x + mean),3) for x in list(lasso.predict(tests))]
		df_test_nan.ix[rf_pre_data.index,c] = rf_pre_data[c]
		print("have been fill:",c)
	test_name = "./clean_data/test.csv"
	df_test_nan.to_csv(test_name,encoding='gbk',index=False)
	return test_name

def fill_train(file_name, base_name):
	base_train = pd.read_csv(base_name,encoding='gbk')
	df_train_nan = pd.read_csv(file_name,encoding='gbk')
	cols = list(df_train_nan.columns)
	with_nan_cols = [k for k,v in zip(list(df_train_nan.columns),list(df_train_nan.isnull().any())) if
	                 v == True and k not in ['id',"date","blood_sugar","blood_sugar_log"]]
	print("have nan's columns: ", with_nan_cols)
	model_fill_name = "./clean_data/model_fill_train.csv"
	for c in cols:
		rf_pre_data = df_train_nan[df_train_nan[c].isnull().values == True]
		if rf_pre_data.shape[0] == 0:
			continue
		no_index = [x for x in list(df_train_nan.index) if x not in list(rf_pre_data.index)]
		rf_no_pre_data = df_train_nan.ix[no_index,:]
		all_rf_data = pd.concat([rf_no_pre_data,rf_pre_data],axis=0)
		no_nan_cols = [k for k,v in zip(list(rf_pre_data.columns),list(rf_pre_data.isnull().any())) if
		               v == False and k not in ['id',"date","blood_sugar","blood_sugar_log"]]
		mean = round((df_train_nan[c].mean() + base_train[c].mean()) / 2,3)
		trains = np.array(base_train[no_nan_cols])
		labels = np.array(base_train[c])
		tests = np.array(rf_pre_data[no_nan_cols])
		# rf = RandomForestRegressor(n_estimators = 200,min_samples_leaf = 20,random_state = 1024,max_depth=6)
		lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=42))
		lasso.fit(trains,labels)
		rf_pre_data[c] = [round((2 * x * mean) / (x + mean),3) for x in list(lasso.predict(tests))]
		df_train_nan.ix[rf_pre_data.index,c] = rf_pre_data[c]
		print("have been fill:",c)
	df_train_nan.to_csv(model_fill_name, encoding='gbk',index=False)
	return model_fill_name
def main():
	pass

if __name__ == '__main__':
	main()