import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

result_pred = pd.read_csv("./sub_xgb_16_2_h.csv")
print("begin process gt10..")
gt10_prob = pd.read_csv("../classification/result/gt10.csv")
gt10_prob.sort_values(by='gt10_prob',inplace=True)
gt10_25 = gt10_prob.tail(25)

result_pred = pd.merge(result_pred,gt10_25,on='id',how='left')
result_pred.fillna(-999,inplace=True)
result_pred = result_pred[result_pred.gt10_prob==-999]
result_pred = result_pred[['id','pred']]
print("finish process gt10...")
print("begin process lt45...")
lt45_prob = pd.read_csv('../classification/result/lt45.csv')
lt45_prob.sort_values(by='lt45_prob',inplace=True)
lt45_50 = lt45_prob.tail(50)
result_pred = pd.merge(result_pred,lt45_50,on='id',how='left')
result_pred.fillna(-999,inplace=True)
result_pred = result_pred[result_pred.lt45_prob==-999]
result_pred = result_pred[['id','pred']]
print("finish process lt45...")
lt45_50['pred'] = 4.5
lt45_50.drop('lt45_prob',axis=1,inplace=True)
gt10_25['pred'] = 10
gt10_25.drop('gt10_prob',axis=1,inplace=True)
print("merge..")
submission = pd.concat([result_pred,lt45_50,gt10_25],axis=0)
submission.sort_values(by='id',inplace=True)
submission["pred"].to_csv('../submission/final_predict_result.csv',index=False, header=False)
submission["pred"].describe()