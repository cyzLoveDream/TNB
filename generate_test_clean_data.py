import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def clean_test_data(file_name):
	print("begin clean test data...")
	df_test = pd.read_csv(file_name,encoding='gbk')
	rename_dict = {"性别": "gender","年龄": "age","体检日期": "date","血糖": "blood_sugar"}
	re_rename_dict = {v: k for k,v in zip(rename_dict.keys(),rename_dict.values())}
	df_test.rename(columns=rename_dict,inplace=True)
	features = [x for x in list(df_test.columns) if x not in ["gender","age","date","blood_sugar","id"]]
	fea_rename = {v: "feature_" + str(k) for k,v in enumerate(features)}
	re_fea_rename = {v: k for k,v in zip(fea_rename.keys(),fea_rename.values())}
	df_test["gender"] = df_test.gender.apply(lambda x: 1 if x == "男" else 0)
	df_test.rename(columns=fea_rename,inplace=True)
	df_test.drop(["feature_15","feature_16","feature_17","feature_18","feature_19"],axis=1,inplace=True)
	re_fea_rename.update(re_rename_dict)
	# df_test.rename(columns=re_fea_rename, inplace=True)
	pre_test_name = './pre_clean_data/pre_test_B.csv'
	df_test.to_csv(pre_test_name,index=False,encoding='gbk')
	print("finish...")
	return pre_test_name
def main():
	
	clean_test_data('./raw_data/d_test_A.csv')

if __name__ == '__main__':
	main()