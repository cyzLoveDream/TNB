import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

old = pd.read_csv("./sub_xgb_16_2_h.csv")
result_pred = pd.read_csv("./sub_lgb_16_1_c_8323.csv",names=["pred"])
result_pred["id"] = old["id"]
print("begin process gt10..")

gt10_prob = pd.read_csv("../classification/result/gt10.csv")
gt10_prob.sort_values(by='gt10_prob',inplace=True)
gt10_prob = gt10_prob.tail(7)
result_pred = pd.merge(result_pred,gt10_prob,on='id',how='left')
result_pred.fillna(-999,inplace=True)
result_pred_sou = result_pred[result_pred.gt10_prob==-999]
result_pred_gt = result_pred[result_pred.gt10_prob != -999]
result_pred = result_pred_sou[['id','pred']]
result_pred_gt["pred"] = result_pred_gt["pred"] * 1.4
result_pred_gt = result_pred_gt[["id","pred"]]

print("begin process lt45...")

lt45_prob = pd.read_csv('../classification/result/lt45.csv')
lt45_prob.sort_values(by='lt45_prob',inplace=True)
lt45 = lt45_prob.tail(20)
result_pred = pd.merge(result_pred,lt45,on='id',how='left')
result_pred.fillna(-999,inplace=True)
result_pred_sor = result_pred[result_pred.lt45_prob==-999]
result_pred_lt = result_pred[result_pred.lt45_prob !=-999]
result_pred = result_pred_sor[['id','pred']]
result_pred_lt["pred"] = result_pred_lt["pred"] * 0.8
result_pred_lt = result_pred_lt[["id","pred"]]

print("merge..")
submission = pd.concat([result_pred,result_pred_lt,result_pred_gt],axis=0)
submission.sort_values(by='id',inplace=True)
submission["pred"].to_csv('../submission/sub_lgb_18_1_c.csv',index=False, header=False)
print(submission["pred"].describe())