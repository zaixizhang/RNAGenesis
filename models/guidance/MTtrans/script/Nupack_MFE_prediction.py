#!/ssd/users/wergillius/.conda/envs/pytorch/bin/python
import os
import sys
import numpy as np
import pandas as pd
import nupack
import PATH
import utils
import scipy
from tqdm import tqdm
tqdm.pandas()

my_model=nupack.Model(material='RNA')
eGFP_seq = utils.eGFP_seq

data_dir= utils.data_dir
csv_name = sys.argv[1]
csv_path = csv_name if os.path.exists(csv_name) else os.path.join(data_dir, csv_name)

assert os.path.exists(csv_path), "csv not found"


df = pd.read_csv(csv_path)
seq_col = 'seq' if 'seq' in df.columns else 'utr'
tr_col = 'rl' if 'rl' in df.columns else 'log_te'

def cor_fun(col):
    pearson_r = scipy.stats.pearsonr(df[col].values,
                df[tr_col].values)[0]
    return pearson_r

nupck_engery_fn = lambda x: nupack.mfe(strands=[x], model=my_model)[0].energy
df['nupack_MFE']  = df[seq_col].progress_apply(nupck_engery_fn)

print("nupack_MFE : {}".format(cor_fun('nupack_MFE')))

# nupck_engery_fn = lambda x: nupack.mfe(strands=[x + eGFP_seq[:10]], model=my_model)[0].energy
# df['eGFP10_nupMFE']  = df[seq_col].progress_apply(nupck_engery_fn)

# print("eGFP10_nupMFE : {}".format(cor_fun('eGFP10_nupMFE')))

# nupck_engery_fn = lambda x: nupack.mfe(strands=[x + eGFP_seq[:50]], model=my_model)[0].energy
# df['eGFP50_nupMFE']  = df[seq_col].progress_apply(nupck_engery_fn)

# print("eGFP50_nupMFE : {}".format(cor_fun('eGFP50_nupMFE')))

df.to_csv(csv_path, index=False)
print(f"saved to {csv_path}")