import os
import sys
import pandas as pd
import numpy as np
import PATH
import torch
import argparse
# 
from models import reader
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP
# 
from sklearn.linear_model import Lasso, Ridge, ElasticNet

#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=str, default='test', help='[train, val, test]')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_cv", type=int, default=1, help='the repeat')
args = parser.parse_args()

config_path = args.config
config = Auto_popen(config_path)
config.kfold_cv = 'train_val'
mu_cv0 = MAP.Maximum_activation_patch(popen=config, which_layer=4,
                                      n_patch=args.n_max_act,
                                      kfold_index=args.kfold_cv)
all_task = config.cycle_set
featmap = mu_cv0.extract_feature_map(task='MPA_H',which_set=2)

### take out the GRU memory to get rl decision chain

# detect change point and find cluster