import os
import sys
import pandas as pd
import numpy as np
import PATH
import torch
import argparse
import h5py
# 
from subprocess import DEVNULL, STDOUT, check_call
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP
# 
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import warnings
warnings.filterwarnings('ignore')


#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=int, default=2, help='train - 0 ,val - 1, test - 2 ')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_cv", type=int, default=1, help='the repeat')
parser.add_argument("-d", "--device", type=str, default='cpu', help='the device to use to extract featmap, digit or cpu')
args = parser.parse_args()

config_path = args.config
save_path = config_path.replace(".ini", "_coef")
config = Auto_popen(config_path)
config.batch_size = 256
config.kfold_cv = 'train_val'
all_task = config.cycle_set

task_channel_effect = {}
task_performance = {}

# path check
if os.path.exists(config_path) and not os.path.exists(save_path):
    os.mkdir(save_path)

for task in all_task:

    # .... format featmap as data ....
    print(f"\n\nevaluating for task: {task}")
    # re-instance the map for each task
    map_task = MAP.Maximum_activation_patch(popen=config, which_layer=4,
                                      n_patch=args.n_max_act,
                                      kfold_index=args.kfold_cv,
                                      device_string=args.device)

    # extract feature map and rl decision chain
    # get X
    model, dataloader= map_task.loading(task=task, which_set=args.set)
    max_seq_len = map_task.df[config.seq_col].apply(len).max()

    X_ls = []
    for Data in dataloader:
        # iter each batch
        x,y = train_val.put_data_to_cuda(Data, map_task.popen,False) 
        X_ls.append(x.detach().cpu().numpy())
    X = np.concatenate(X_ls, axis=0)
    X = np.transpose(X, (0,2,1))
    
    attribute = map_task.get_input_grad(task=task, focus=False, fm=X, starting_layer=0)
    X = X[:, :, -1*max_seq_len:]
    attribute = attribute[:,:, -1*max_seq_len:]

    x_npz_path = os.path.join(save_path, f"{task}_set{args.set}_x.npz")
    grad_npz_path = os.path.join(save_path, f"{task}_set{args.set}_grad.npz")
    np.savez(x_npz_path, X)
    np.savez(grad_npz_path, attribute)

    
    
    for n_seqlet in range(500, 4000, 500):
        outfile = os.path.join(save_path, f"{task}_n{n_seqlet}_tfmodisco.h5")
        command = f"modisco motifs -s {x_npz_path} -a {grad_npz_path} -n {n_seqlet} -o {outfile}"
        
        check_call(command.split(), stdout=DEVNULL, stderr=STDOUT)

        with h5py.File(outfile, "r") as f:
            print(f.keys())
            if 'pos_patterns' in f.keys():
                pos_motif = f['pos_patterns'] 
                n_pos = len(pos_motif.keys())
            else:
                n_pos = -1

            if 'neg_patterns' in f.keys():
                neg_motif = f['neg_patterns'] 
                n_neg = len(neg_motif.keys())
            else:
                n_neg = -1
            f.close()

        print(f"discover {n_pos} pos pattern and {n_neg} neg patterns for n = {n_seqlet}")
    print(f'done for {task}')