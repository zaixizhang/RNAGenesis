import os
import sys
import numpy as np
import pandas as pd
import scipy

from tqdm import tqdm
    
import pickle 
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor, GradientBoostingRegressor


# define model

data_name = sys.argv[1]
k = str(sys.argv[2])

# assert data_name in ['muscle', 'Andrev2015', 'pc3', 'G65778', 'G52809']
assert data_name in ['293T','PCR3']

# read data
f = open(f'log/RF_zhang/{data_name}_chr{k}.log','w')
# f = open(f'log/{data_name}_chr{k}_noutr.log','w')
sparse_mat = np.load(f"/data/users/wergillius/UTR_VAE/5UTR_Optimizer/output/RP_{data_name}_covar_sparse_feature.npy",allow_pickle=True)
df  = pd.read_csv(f"/data/users/wergillius/UTR_VAE/MTtrans/RP_{data_name}_covar.csv")

# corvariate = ['utr_maxmfe', 'utr_meanmfe', 'utr_minmfe', 'cds_maxmfe', 'cds_meanmfe', 'cds_minmfe', 'utr3_maxmfe', 'utr3_meanmfe', 'utr3_minmfe', 'last_uAUG_site', 'AUG_inframe', 'utr_num_uAUG', 'utr_GC', 'utr_len', 'cds_GC', 'cds_len', 'utr3_GC', 'utr3_len', 'freq_AAA', 'freq_AAG', 'freq_AAC', 'freq_AAT', 'freq_AGA', 'freq_AGG', 'freq_AGC', 'freq_AGT', 'freq_ACA', 'freq_ACG', 'freq_ACC', 'freq_ACT', 'freq_ATA', 'freq_ATG', 'freq_ATC', 'freq_ATT', 'freq_GAA', 'freq_GAG', 'freq_GAC', 'freq_GAT', 'freq_GGA', 'freq_GGG', 'freq_GGC', 'freq_GGT', 'freq_GCA', 'freq_GCG', 'freq_GCC', 'freq_GCT', 'freq_GTA', 'freq_GTG', 'freq_GTC', 'freq_GTT', 'freq_CAA', 'freq_CAG', 'freq_CAC', 'freq_CAT', 'freq_CGA', 'freq_CGG', 'freq_CGC', 'freq_CGT', 'freq_CCA', 'freq_CCG', 'freq_CCC', 'freq_CCT', 'freq_CTA', 'freq_CTG', 'freq_CTC', 'freq_CTT', 'freq_TAC', 'freq_TAT', 'freq_TGG', 'freq_TGC', 'freq_TGT', 'freq_TCA', 'freq_TCG', 'freq_TCC', 'freq_TCT', 'freq_TTA', 'freq_TTG', 'freq_TTC', 'freq_TTT', 'freq_AAA', 'freq_AAG', 'freq_AAC', 'freq_AAT', 'freq_AGA', 'freq_AGG', 'freq_AGC', 'freq_AGT', 'freq_ACA', 'freq_ACG', 'freq_ACC', 'freq_ACT', 'freq_ATA', 'freq_ATG', 'freq_ATC', 'freq_ATT', 'freq_GAA', 'freq_GAG', 'freq_GAC', 'freq_GAT', 'freq_GGA', 'freq_GGG', 'freq_GGC', 'freq_GGT', 'freq_GCA', 'freq_GCG', 'freq_GCC', 'freq_GCT', 'freq_GTA', 'freq_GTG', 'freq_GTC', 'freq_GTT', 'freq_CAA', 'freq_CAG', 'freq_CAC', 'freq_CAT', 'freq_CGA', 'freq_CGG', 'freq_CGC', 'freq_CGT', 'freq_CCA', 'freq_CCG', 'freq_CCC', 'freq_CCT', 'freq_CTA', 'freq_CTG', 'freq_CTC', 'freq_CTT', 'freq_TAC', 'freq_TAT', 'freq_TGG', 'freq_TGC', 'freq_TGT', 'freq_TCA', 'freq_TCG', 'freq_TCC', 'freq_TCT', 'freq_TTA', 'freq_TTG', 'freq_TTC', 'freq_TTT']
corvariate = ['cds_maxmfe', 'cds_meanmfe', 'cds_minmfe', 'utr3_maxmfe', 'utr3_meanmfe', 'utr3_minmfe', 'cds_GC', 'cds_len', 'utr3_GC', 'utr3_len']

X_utr = sparse_mat[:,1:]
X_covar = df[corvariate].values
# X = X_covar
X = np.concatenate([X_utr, X_covar], axis=1)
y = df['log_te'].values

train_id = df.query("`chromosome` != @k").index
test_id = df.query("`chromosome` == @k").index
x_train, x_test = X[train_id] , X[test_id]
y_train, y_test = y[train_id] , y[test_id]

f.write(f'Data Size : {x_train.shape}\n')
model = RandomForestRegressor(n_estimators=50,n_jobs=20).fit(x_train, y_train)

y_pred = model.predict(x_test)
f.write(f' r2 : {model.score(x_test, y_test)} \n')
f.write(f' spr : {stats.spearmanr(y_pred, y_test)[0]} \n')
f.write(f' pr : {stats.pearsonr(y_pred, y_test)[0]} \n')

# model_save_path = f'/home/wergillius/Project/5UTR_Optimizer/data/{data_name}/fewer_log_control.model'
# filehandler = open(model_save_path, 'wb') 
# pickle.dump(model_dict, filehandler)
f.close()