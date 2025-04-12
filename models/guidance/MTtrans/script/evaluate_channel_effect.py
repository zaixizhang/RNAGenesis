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
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')


#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=int, default=2, help='train - 0 ,val - 1, test - 2 ')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_cv", type=int, default=1, help='the repeat')
parser.add_argument("-t", "--task", type=str, default='regression', help='either regression or classification')
parser.add_argument("-d", "--device", type=str, default='cpu', help='the device to use to extract featmap, digit or cpu')
args = parser.parse_args()

config_path = args.config
save_path = config_path.replace(".ini", "_coef")
config = Auto_popen(config_path)
config.batch_size = 256
# config.kfold_cv = 'train_val'
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
    featmap = map_task.extract_feature_map(task=task, which_set=args.set)
    cum_rl_trend = map_task.cumulative_rl_decision(task=task, which_set=args.set)

    # truncate the featmap and rl trend according to sequence length
    max_seq_len = map_task.df[config.seq_col].apply(len).max()
    to_stay = max_seq_len // np.product(map_task.strides) +1
    trunc_start = featmap.shape[2] - to_stay
    featmap = featmap[:,:,trunc_start:]
    cum_rl_trend = cum_rl_trend[:,trunc_start:]

    # construct input for linear regression
    n_sample,n_channel,n_posi = featmap.shape
    
    X = featmap.reshape(n_sample,-1)
    Y = map_task.Y_ls.flatten()

    # .... regression ....
    if args.task == 'regression':
        L1 = LassoCV(alphas=np.linspace(2e-3, 0.1, 49))
        L2 = RidgeCV(alphas=np.linspace(0.001, 0.101, 20))
        elastic = ElasticNetCV(alphas=np.linspace(2e-3, 0.1, 49), n_jobs=10)    
        models = [L1, L2, elastic]
        model_names = ['Lasso', 'Ridge', 'Elastic']
 
    else:
        LR = LogisticRegressionCV(n_jobs=10)
        # L1 = LogisticRegressionCV(n_jobs=10,penalty='l1', solver='saga')
        # elastic = LogisticRegressionCV(n_jobs=10,penalty='elasticnet', solver='saga', l1_ratios=np.linspace(0.0, 0.5, 10))
        models = [LR]
        model_names = ['Logistic']
 

    for model,name in zip(models, model_names):
        print(f"\nregressing {name}..")
        model.fit(X,Y)
        r2 = model.score(X,Y)  # will be
        sparsity = np.sum(model.coef_==0) / X.shape[1] *100 
        try:
            alpha = model.alpha_
        except:
            alpha = 0.0

        print(f"{name} with optimal alpha {alpha:.5f}, r2/acc {r2:.3f} , zero coeff {sparsity:.1f}%")
    
        # save df
        effect = model.coef_.reshape(n_channel,-1)
        fullcoef_df = pd.DataFrame(effect, columns=[f"{name}_posi_"+str(trunc_start+i) for i in range(to_stay)])
        fullcoef_df.to_csv( os.path.join(save_path , f"{task}_{name}_coef.csv"), index=False)
        task_channel_effect[f"{task}_{name}"] = effect.mean(axis=1)

        task_performance[f"{task}_{name}"] = [alpha, r2, sparsity, model.coef_.max(), model.coef_.min()]  


all_effect = pd.DataFrame(task_channel_effect)
all_effect.to_csv(os.path.join(save_path, "all_task_mean_effect.csv"), index=False)

report_df = pd.DataFrame(task_performance)
report_df.index = ['optim_alpha','r2', 'zero_pctg', 'max_coef', 'min_coef']
report_df.to_csv(os.path.join(save_path, "regression_report.csv"), index=False)
