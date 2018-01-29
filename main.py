from generate_train_clean_data import clean_train_data
from generate_test_clean_data import clean_test_data
from fill_value import *
import time

def clean_data():
	"""
	base_train_time: 基本的训练集，没有任何空值
	train_with_nan_name: 训练集中有空值的数据
	pre_test_name: 测试集中有空值的数据
	model_file_name: 训练集中的空值使用随机森林模型(使用base_train训练) + 调和均值填充
	tests: 测试集的空值使用随机森林模型（使用base_train训练） + 调和均值填充
	:return:
	"""
	train_name = './raw_data/d_train.csv'
	test_name = './raw_data/d_test_B_20180128.csv'
	now = time.time()
	base_train_name,train_with_nan_name = clean_train_data(train_name)
	pre_test_name = clean_test_data(test_name)
	print("begin fill train...")
	model_fill_name = fill_train(train_with_nan_name,base_train_name)
	print("begin fill test...")
	tests = fill_test(pre_test_name,base_train_name)
	print("finish fill data in time: ",time.time() - now)
	return base_train_name, model_fill_name, tests

def generate_1(all_data):
	all_data["feature_0_is_normal"] = all_data["feature_0"].apply(lambda x: 0 if x < 10 else (2 if x > 40 else 1))
	all_data["feature_1_is_normal"] = all_data["feature_1"].apply(lambda x: 0 if x < 10 else (2 if x > 40 else 1))
	all_data["feature_1_feature_0"] = round(all_data["feature_0"] / all_data['feature_1'],3)
	all_data["feature_1_feature_0_is_normal"] = all_data["feature_1_feature_0"].apply(lambda x: 0 if x > 1 else 1)
	all_data["feature_2_is_normal"] = all_data["feature_2"].apply(lambda x: 0 if x < 40 else (2 if x > 150 else 1))
	all_data["feature_3_is_normal"] = all_data[["feature_3","gender"]].apply(lambda x:
	                                                                         (0 if x[0] < 11 else(
	                                                                         2 if x[0] > 50 else 1)) if x[1] == 1 else (
	                                                                         3 if x[0] < 7 else(5 if x[0] > 32 else 4)),
	                                                                         axis=1)
	all_data["feature_4_is_normal"] = all_data["feature_4"].apply(lambda x: 0 if x < 60 else (2 if x > 80 else 1))
	all_data["feature_5_is_normal"] = all_data["feature_5"].apply(lambda x: 0 if x < 40 else (2 if x > 55 else 1))
	all_data["feature_6_is_normal"] = all_data["feature_6"].apply(lambda x: 0 if x < 20 else (2 if x > 30 else 1))
	all_data["feature_8_is_normal"] = all_data["feature_8"].apply(lambda x: 0 if x < 0.42 else (2 if x > 1.82 else 1))
	all_data["feature_9_is_normal"] = all_data["feature_9"].apply(lambda x: 0 if x < 2.9 else (2 if x > 6 else 1))
	all_data["feature_10_is_normal"] = all_data["feature_10"].apply(lambda x: 0 if x < 1.16 else (2 if x > 1.55 else 1))
	all_data["feature_10_gender_is_normal"] = all_data[["feature_10","gender"]].apply(lambda x:
	                                                                                  (0 if x[0] < 1.1 else 1) if x[
		                                                                                                              1] == 1 else (
	                                                                                  2 if x[0] < 1.2 else 3),
	                                                                                  axis=1)
	all_data["feature_11_is_normal"] = all_data["feature_11"].apply(lambda x: 0 if x < 2.84 else (2 if x > 3.12 else 1))
	all_data["feature_10_feature_11"] = round(all_data["feature_11"] / all_data['feature_10'],3)
	all_data["feature_12_is_normal"] = all_data["feature_12"].apply(lambda x: 0 if x < 1.7 else (2 if x > 8.3 else 1))
	all_data["feature_13_is_normal"] = all_data[["feature_13","gender"]].apply(lambda x:
	                                                                           (0 if x[0] < 53 else(
	                                                                           2 if x[0] > 106 else 1)) if x[
		                                                                                                       1] == 1 else (
	                                                                           3 if x[0] < 44 else(
	                                                                           5 if x[0] > 97 else 4)),
	                                                                           axis=1)
	all_data["feature_14_is_normal"] = all_data[["feature_14","gender"]].apply(lambda x:
	                                                                           (0 if x[0] < 150 else(
	                                                                           2 if x[0] > 416 else 1)) if x[
		                                                                                                       1] == 1 else (
	                                                                           3 if x[0] < 89 else(
	                                                                           5 if x[0] > 357 else 4)),
	                                                                           axis=1)
	all_data["feature_14_gender_is_normal"] = all_data[["feature_14","gender"]].apply(lambda x:
	                                                                                  (0 if x[0] > 420 else 1) if x[
		                                                                                                              1] == 1 else (
	                                                                                  2 if x[0] > 350 else 3),
	                                                                                  axis=1)
	all_data["feature_20_is_normal"] = all_data["feature_20"].apply(lambda x: 0 if x < 4 else (2 if x > 10 else 1))
	all_data["feature_21_is_normal"] = all_data[["feature_21","gender"]].apply(lambda x:
	                                                                           (0 if x[0] < 4.0 else(
	                                                                           2 if x[0] > 5.5 else 1)) if x[
		                                                                                                       1] == 1 else (
	                                                                           3 if x[0] < 3.5 else(
	                                                                           5 if x[0] > 5.0 else 4)),
	                                                                           axis=1)
	all_data["feature_22_is_normal"] = all_data[["feature_22","gender"]].apply(lambda x:
	                                                                           (0 if x[0] < 120 else(
	                                                                           2 if x[0] > 160 else 1)) if x[
		                                                                                                       1] == 1 else (
	                                                                           3 if x[0] < 110 else(
	                                                                           5 if x[0] > 150 else 4)),
	                                                                           axis=1)
	all_data["feature_23_is_normal"] = all_data[["feature_23","gender"]].apply(lambda x:
	                                                                           (0 if x[0] < 0.4 else(
	                                                                           2 if x[0] > 0.5 else 1)) if x[
		                                                                                                       1] == 1 else (
	                                                                           3 if x[0] < 0.37 else(
	                                                                           5 if x[0] > 0.48 else 4)),
	                                                                           axis=1)
	all_data["feature_24_is_normal"] = all_data["feature_24"].apply(lambda x: 0 if x < 80 else (2 if x > 100 else 1))
	all_data["feature_25_is_normal"] = all_data["feature_25"].apply(lambda x: 0 if x < 27 else (2 if x > 34 else 1))
	all_data["feature_26_is_normal"] = all_data["feature_26"].apply(lambda x: 0 if x < 320 else (2 if x > 360 else 1))
	all_data["feature_27_is_normal"] = all_data["feature_27"].apply(lambda x: 0 if x < 11.5 else (2 if x > 14.5 else 1))
	all_data["feature_28_is_normal"] = all_data["feature_28"].apply(lambda x: 0 if x < 100 else (2 if x > 300 else 1))
	all_data["feature_29_is_normal"] = all_data["feature_29"].apply(lambda x: 0 if x < 9 else (2 if x > 13 else 1))
	all_data["feature_30_is_normal"] = all_data["feature_30"].apply(lambda x: 0 if x < 9 else (2 if x > 17 else 1))
	all_data["feature_31_is_normal"] = all_data["feature_31"].apply(lambda x: 0 if x < 0.13 else (2 if x > 0.23 else 1))
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
	return all_data

def generate_2(all_data):
	all_data["feature_28_feature_31"] = all_data["feature_28"] * all_data["feature_31"]
	all_data["ATSs"] = all_data["feature_0"] * 0.8
	all_data["ATSm"] = all_data["feature_0"] * 0.2
	all_data["feature_0_feature_1"] = all_data["feature_0"] + all_data["feature_1"]
	all_data["feature_4_less_60"] = all_data["feature_4"].apply(lambda x: 1 if x < 60 else 0)
	all_data["feature_5_less_25"] = all_data["feature_5"].apply(lambda x: 1 if x < 25 else 0)
	all_data["feature_4_feature_5"] = all_data["feature_4"] + all_data["feature_5"]
	all_data["feature_4_feature_6"] = all_data["feature_4"] + all_data["feature_6"]
	return all_data

def generate_feature(base_train_name,model_fill_name,tests_name):
	print("begin generate feature...")
	trains = pd.read_csv(base_train_name,encoding="gbk")
	test = pd.read_csv(tests_name,encoding='gbk')
	vals = pd.read_csv(model_fill_name,encoding='gbk')
	all_data = pd.concat([trains,vals,test],axis=0,ignore_index=True)
	all_data = generate_1(all_data)
	train = all_data[:trains.shape[0] + vals.shape[0]]
	tests = all_data[trains.shape[0] + vals.shape[0]:]
	cate_fea = ['feature_0_is_normal',
	            'feature_1_is_normal',
	            'feature_1_feature_0_is_normal','feature_2_is_normal',
	            'feature_3_is_normal','feature_4_is_normal','feature_5_is_normal',
	            'feature_6_is_normal','feature_8_is_normal','feature_9_is_normal',
	            'feature_10_is_normal','feature_10_gender_is_normal',
	            'feature_11_is_normal','feature_12_is_normal',
	            'feature_13_is_normal','feature_14_is_normal',
	            'feature_14_gender_is_normal','feature_20_is_normal',
	            'feature_21_is_normal','feature_22_is_normal','feature_23_is_normal',
	            'feature_24_is_normal','feature_25_is_normal','feature_26_is_normal',
	            'feature_27_is_normal','feature_28_is_normal','feature_29_is_normal',
	            'feature_30_is_normal','feature_31_is_normal','feature_32_is_normal',"feature_1_feature_0",
	            'feature_10_feature_11',
	            'feature_20_feature_32','feature_33_is_normal',
	            'feature_20_feature_33','feature_34_is_normal',
	            'feature_20_feature_34','feature_35_is_normal',
	            'feature_20_feature_35','feature_36_is_normal',
	            'feature_20_feature_36',"id"]
	train[cate_fea].to_csv("./raw_data/fea_train.csv",index=False)
	tests[cate_fea].to_csv("./raw_data/fea_test_B.csv",index=False)
	all_data = generate_2(all_data)
	all_data = all_data[["id",'feature_28_feature_31','ATSs','ATSm',
	                     'feature_0_feature_1','feature_4_less_60','feature_5_less_25',
	                     'feature_4_feature_5','feature_4_feature_6']]
	train = all_data[:trains.shape[0] + vals.shape[0]]
	tests = all_data[trains.shape[0] + vals.shape[0]:]
	train.to_csv("./raw_data/fea_train_1.csv",index=False)
	tests.to_csv("./raw_data/fea_test_B_1.csv",index=False)
	print("finish generate feature...")
def main():
	base_train_name,model_fill_name,tests = clean_data()
	# generate_feature(base_train_name, model_fill_name, tests)
	pass

if __name__ == '__main__':
	main()