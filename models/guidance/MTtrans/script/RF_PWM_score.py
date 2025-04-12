import os
import sys
import PATH 
import numpy as np
import pandas as pd
import scipy
import argparse
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
from scipy import stats
from models import reader
from scipy.stats import randint
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier, GradientBoostingClassifier


parser = argparse.ArgumentParser("use the PWM score to generate ")
parser.add_argument("-X","--npy",type=str, help="the abs path of PWM scores for each sequences")
parser.add_argument("-D","--df",type=str, help="the abs path of the sequence dataframe")
parser.add_argument("-N","--n_estimators",type=int, help="the number of estimators")
parser.add_argument("-K","--kfold_index",type=int, help="the random seed of splited fold")
parser.add_argument("-L","--label_col",type=str, help="the dataframe col that contain the label")
args = parser.parse_args()
X_path  = args.npy
df_path = args.df
K  = args.kfold_index
N =  args.n_estimators
Label = args.label_col

X = np.load(X_path)
Seq_df = pd.read_csv(df_path)
if Seq_df[Label].nunique != 2:
    subset_index = Seq_df[Label].isin([0,1])
    Seq_df = Seq_df[subset_index]
    X = X[subset_index]

Y = Seq_df[Label].values


df_ls = reader.KFold_df_split(Seq_df.reset_index(), K)
train_index, val_index, test_index = [df.index for df in df_ls]
trainval_index  = np.concatenate([train_index, val_index])

X_train, Y_train = X[trainval_index], Y[trainval_index]
X_test, Y_test = X[test_index], Y[test_index]

RF = RandomForestClassifier(n_jobs=20,n_estimators=N)
AdaB = AdaBoostClassifier(n_estimators=N, learning_rate=1)
GB = GradientBoostingClassifier(n_estimators=N)
LR = LogisticRegressionCV(n_jobs=10, max_iter=500)
# CatB = CatBoostClassifier(iterations=N*10, task_type="GPU",devices='0:2', od_type='Iter', random_seed=42)
models = [RF,AdaB, GB, LR]
names = ['RF', "AdaBoost", "GradientBoost", "LinearRegression"]
models = [ RF, LR]
names = ["RF","LogisticRegression"]

# parameters = {
#     "depth" : randint(4,10),
#     "iterations": randint(200,1000) 
# }
# randm = RandomizedSearchCV(estimator=CatB, param_distributions=parameters, cv=5, n_iter=10, n_jobs=20)

# models = [randm]
# names = ["RandomizedSearchCV"]

pred_df = pd.DataFrame()
pred_df["test_index"] = test_index
pred_df["Y_true"] = Y_test

coef_df = pd.DataFrame()

save_dir = f'log/RandomForest/{os.path.basename(X_path)}'.replace(".npy","")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f = open(os.path.join(save_dir,f'{Label}_{K}_n{N}.log'),'w')

for model, model_name in zip(models, names):
    
    if isinstance(model, CatBoostClassifier):
        model.fit(X[train_index], Y[train_index], verbose=False,
                   eval_set = Pool(X[val_index], Y[val_index]))
    elif isinstance(model, RandomizedSearchCV):
        model.fit(X_train, Y_train)
    else:
        model.fit(X_train, Y_train)

    if model_name == "RF":
        importance = model.feature_importances_
        coef_df[f'{model_name}_feat_imp'] = importance
    elif model_name == "LogisticRegression":
        coef_df[f'{model_name}_coef'] = model.coef_.flatten()
    

    Y_pred = model.predict(X_test)

    pred_df[f'{model_name}_Y_pred'] = Y_pred
    pred_df["test_index"] = test_index

    acc = model.score(X_test, Y_test)
    F1 = f1_score(Y_test,Y_pred>0.5)
    auroc = roc_auc_score(Y_test,Y_pred)

    f.write(f'\n {model_name} \n')
    f.write(f' ACC : {acc} \n')
    f.write(f' F1 : {F1} \n')
    f.write(f' AUROC : {auroc} \n')

for featimp_cutoff in [0.5,0.6,0.8,0.9]:
    thres = np.quantile(importance, q=featimp_cutoff)
    important_feat = importance > thres # important feats
    LR_feat_select = LogisticRegressionCV(n_jobs=10, max_iter=500).fit(X_train[:,important_feat], Y_train)
    Y_pred = LR_feat_select.predict(X_test[:,important_feat])

    acc = LR_feat_select.score(X_test[:,important_feat], Y_test)
    F1 = f1_score(Y_test,Y_pred>0.5)
    auroc = roc_auc_score(Y_test,Y_pred)

    coef_col = f'LR_feat{featimp_cutoff}_coef'
    coef_df[coef_col] = np.full((256,),np.nan)
    coef_df.loc[important_feat,coef_col] = LR_feat_select.coef_.flatten() 

    f.write(f'\n LR using top {featimp_cutoff} most important features  \n')
    f.write(f' ACC : {acc} \n')
    f.write(f' F1 : {F1} \n')
    f.write(f' AUROC : {auroc} \n')

pred_df.to_csv(os.path.join(save_dir,f"{Label}_{K}_pred_n{N}.csv"), index=False)
coef_df.to_csv(os.path.join(save_dir,f"{Label}_{K}_coef_n{N}.csv"), index=False)
f.close()

# ["-X", "/data/users/wergillius/UTR_VAE/Non-Viral_Gene_Therapies_GSE176581/merged_bin_data_3M_Hinfo_score.npy",
#     "-D",  "/data/users/wergillius/UTR_VAE/Non-Viral_Gene_Therapies_GSE176581/merged_bin_data.csv",
#     "-K",  "0",
#     "-N",  "400",
#     "-L", "clf_label_30%"
# ]
# ["-X","/data/users/wergillius/UTR_VAE/Alan_dataset/AlanAll_10pctg_3M_Vinfo_score.npy",
#     "-D",  "/data/users/wergillius/UTR_VAE/Alan_dataset/AlanAll_binary_10pctg.csv",
#     "-K",  "1",
#     "-N",  "400",
#     "-L", "Binary_10pc"
# ]