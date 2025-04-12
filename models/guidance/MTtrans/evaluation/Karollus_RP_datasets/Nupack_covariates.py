#!/ssd/users/wergillius/.conda/envs/pytorch/bin/python
import os
import sys
import numpy as np
import pandas as pd
import nupack
import PATH
import utils
import scipy
import time
from tqdm import tqdm
tqdm.pandas()

my_model=nupack.Model(material='RNA')
data_dir= utils.data_dir
csv_path = sys.argv[1]

assert os.path.exists(csv_path), "csv not found"

df = pd.read_csv(csv_path)

nupck_engery_fn = lambda x: nupack.mfe(strands=[x], model=my_model)[0].energy

def sliding_window_mfe(seq, window_size=100):
    """
    statistics of local mfe
    """
    mfe_all_windows = []
    for i in range(0, len(seq), window_size):
        mfe_all_windows.append(
            nupck_engery_fn(seq[i:i+window_size])
                              )
    max_mfe = np.max(mfe_all_windows)
    mean_mfe = np.mean(mfe_all_windows)
    min_mfe = np.min(mfe_all_windows)
    return max_mfe, mean_mfe, min_mfe

for region in ['utr', 'cds', '3utr']:
    start_time = time.time()
    
    slideWindow_out = df[region].apply(sliding_window_mfe)
    array_out = np.stack(slideWindow_out.values.tolist())
    
    df[f'{region}_maxmfe'] = array_out[:,0]
    df[f'{region}_meanmfe'] = array_out[:,1]
    df[f'{region}_minmfe'] = array_out[:,2]
    
    run_time = (time.time() - start_time) / 1000
    print(f"MFE for {region} computed, taking {run_time} s")
    
df.to_csv(csv_path, index=False)
print(f"saved to {csv_path}")