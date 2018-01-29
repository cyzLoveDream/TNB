import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_train_data(file_name):
	# read data
	df_data = pd.read_csv(file_name,encoding="gbk")
	df_a = pd.read_csv("./raw_data/d_test_A.csv",encoding="gbk")
	df_a_res = pd.read_csv("./raw_data/d_answer_a_20180128.csv",names=["血糖"])
	df_a["血糖"] = df_a_res["血糖"].values
	df_data = pd.concat([df_data, df_a],axis=0,ignore_index=True)
	print("have been read data, the data's shape is: ",df_data.shape)
	# rename clumns's name
	rename_dict = {"性别": "gender","年龄": "age","体检日期": "date","血糖": "blood_sugar"}
	re_rename_dict = {v: k for k,v in zip(rename_dict.keys(),rename_dict.values())}
	df_data.rename(columns=rename_dict,inplace=True)
	features = [x for x in list(df_data.columns) if x not in ["gender","age","date","blood_sugar","id"]]
	fea_rename = {v: "feature_" + str(k) for k,v in enumerate(features)}
	re_fea_rename = {v: k for k,v in zip(fea_rename.keys(),fea_rename.values())}
	df_data["gender"] = df_data.gender.apply(lambda x: 1 if x == "男" else 0)
	df_data.rename(columns=fea_rename,inplace=True)
	# re_cols = re_rename_dict.update(re_fea_rename)
	print("begin clean data...")
	df_data.drop(df_data[df_data["age"] >= 86].index,inplace=True)
	df_data.drop(["feature_15","feature_16","feature_17","feature_18","feature_19"],axis=1,inplace=True)
	print("finish clean data...")
	# df_data.rename(columns=re_cols,inplace=True)
	df_data_without_nan = df_data.dropna(axis=0,how='any')
	base_train_name = './clean_data/base_train.csv'
	df_data_without_nan.to_csv(base_train_name,encoding='gbk',index=False)
	no_index = [x for x in list(df_data.index) if x not in list(df_data_without_nan.index)]
	train_with_nan_name = "./pre_clean_data/train_with_nan.csv"
	df_data.ix[no_index,:].to_csv(train_with_nan_name,encoding="gbk",index=False)
	return base_train_name, train_with_nan_name
def main():
	clean_train_data('./raw_data/d_train.csv')
if __name__ == '__main__':
	main()