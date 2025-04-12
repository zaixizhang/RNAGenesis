import os
import sys
import pandas as pd
import numpy as np
import PATH
import torch
import argparse
import seaborn as sns
# 
from models import reader
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP
# 
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import warnings
from scipy.cluster import hierarchy
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')


#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=int, default=2, help='train - 0 ,val - 1, test - 2 ')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_cv", type=int, default=1, help='the repeat')
parser.add_argument("-d", "--device", type=str, default='cpu', help='the device to use to extract featmap, digit or cpu')
args = parser.parse_args()

# args = parser.parse_args(["-c","/ssd/users/wergillius/Project/MTtrans/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini",
#     "-s","2",
#     "-d", "1"])

config_path = args.config
save_path = config_path.replace(".ini", "_coef")
config = Auto_popen(config_path)
config.batch_size = 256
config.shuffle = False
config.kfold_cv = 'train_val'
all_task = config.cycle_set

task_channel_effect = {}
task_performance = {}

# path check
if os.path.exists(config_path) and not os.path.exists(save_path):
    os.mkdir(save_path)

saved_pdf =os.path.join(save_path, 'changepoint_actmap.pdf')
pp = PdfPages(saved_pdf)
channel_cluster_task = {}
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
    seq_len_ls = map_task.df[config.seq_col].apply(len)
    total_stride = np.product(map_task.strides)
    to_stay = seq_len_ls//total_stride - 4
    trunc_start = featmap.shape[2] - to_stay

    # if max_seq_len == 50:
    #     trunc_start = -12


    # find the low rl sequences
    rl_pred = cum_rl_trend[:,-1]
    threshold = np.quantile(rl_pred, [0.05, 0.95])

    low_rl = rl_pred < threshold[0]
    high_rl = rl_pred > threshold[1]

    # subset find low rl change point feature
    lowrl_rl_chain = cum_rl_trend[low_rl]
    lowrl_ft = featmap[low_rl]
    lowrl_detect_region = [slice(s,None) for s in trunc_start[low_rl]]
    changepoint_map = map_task.retrieve_featmap_at_changepoint(lowrl_ft, lowrl_rl_chain, 
                                threshold=-1, direction='less', detect_region=lowrl_detect_region)
    print(changepoint_map.shape)

    # sample high rl feature
    highrl_rl_chain = cum_rl_trend[high_rl]
    highrl_ft = featmap[high_rl]
    highrl_detect_region = [slice(s,None) for s in trunc_start[high_rl]]
    background_map = map_task.retrieve_featmap_at_changepoint(highrl_ft, highrl_rl_chain, 
                                threshold=0.5, direction='greater', detect_region=highrl_detect_region)
    
    # and then subsample
    n_background = background_map.shape[0]
    n_foreground = changepoint_map.shape[0]
    if n_background > n_foreground:
        downsample_seed = np.random.choice(np.arange(0,n_background), size=n_foreground)
        background_map = background_map[downsample_seed]
        n_background = background_map.shape[0]
    
    # concate and order two group
    row_colors = ["#E09832"]*n_background + ["#192D48"]*n_foreground
    all_act=np.concatenate([background_map, changepoint_map], axis=0)
    feat_norm_act = (all_act - all_act.min(axis=0)) / (all_act.max(axis=0) - all_act.min(axis=0))


    # Visualization
    g=sns.clustermap(feat_norm_act, row_colors=row_colors);

    # 'array', 'axis', 'calculate_dendrogram', 'calculated_linkage', 'data', 
    # 'dendrogram', 'dependent_coord', 'independent_coord', 'label', 'linkage', 
    # 'method', 'metric', 'plot', 'reordered_ind', 'rotate', 'shape', 'xlabel', 
    # 'xticklabels', 'xticks', 'ylabel', 'yticklabels', 'yticks'

    
    Z_col=g.dendrogram_col.linkage
    thres = 0.8*max(Z_col[:,2])
    R = hierarchy.dendrogram(Z_col ,color_threshold=thres, truncate_mode=None,
        above_threshold_color='#AAAAAA', p=10, orientation='top',ax=g.ax_col_dendrogram);
    # R['leaves']
    # R['ivl']

    Z_row=g.dendrogram_row.linkage
    thres = 0.8*max(Z_row[:,2])
    R2 = hierarchy.dendrogram(Z_row ,color_threshold=thres, truncate_mode=None, orientation='left',
        above_threshold_color='#AAAAAA', p=10, ax=g.ax_row_dendrogram);
    g.ax_row_dendrogram.invert_yaxis()
    g.figure.suptitle(task)

    pp.savefig(g.figure)

    channel_cluster_task[task] = pd.DataFrame(dict(zip(R['leaves'],R['leaves_color_list'])))
    print("No..")
pp.close()
channel_cluster_df = pd.DataFrame(channel_cluster_task)
saved_csv = os.path.join(save_path, "changepoint_channel_cluster.csv")
channel_cluster_df.to_csv(saved_csv, index=False)
print("==DONE==")
print(f"result save to {saved_pdf}\n \t\t {saved_csv}")

