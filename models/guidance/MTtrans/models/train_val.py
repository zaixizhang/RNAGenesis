import os
import sys
import time
import numpy as np
import pandas as pd
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import torch
from scipy.stats import pearsonr, spearmanr
from torch import optim
from sklearn.metrics import r2_score, f1_score, roc_auc_score
from models.ScheduleOptimizer import ScheduledOptim , find_lr
from models.loss import Dynamic_Task_Priority as DTP

def train(dataloader,model,optimizer,popen,epoch,lr=None, verbose=True):

    logger = logging.getLogger("VAE")
    loader_len = len(dataloader)       # number of iteration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    model.train()
    verbose_list=[]
    verbose_df = pd.DataFrame()
    for idx,data in enumerate(dataloader):
        
        X,Y = put_data_to_cuda(data,popen,require_grad=True)
        optimizer.zero_grad()
        
        out = model(X)
       
        loss_dict = model.compute_loss(out,X,Y,popen)
        loss = loss_dict['Total']
        acc_dict = model.compute_acc(out,X,Y,popen)
        loss_dict.update(acc_dict) # adding the acc ditc into loss dict
         
        loss.backward()        
        optimizer.step()
        
        # ====== update lr =======
        if type(optimizer) == ScheduledOptim:
            lr = optimizer.update_learning_rate()      # see model.optim.py
            loss_dict["lr"]=lr
        elif popen.optimizer == 'Adam':
            lr = optimizer.param_groups[0]['lr']
        if popen.loss_schema != 'constant':
            popen.chimerla_weight = popen.loss_schedualer._update(loss_dict)
            for t in popen.tasks:
                loss_dict[popen.loss_schema+"_wt_"+t] = popen.chimerla_weight[t]
                
        with torch.no_grad():   
            loss_dict= utils.clean_value_dict(loss_dict)
            verbose_list.append(loss_dict)
       
        # ======== verbose ========
        # record result 5 times for a epoch
        if verbose:
            if idx % int(loader_len/5) == 0:
                
                # plot that in loss dict
                loss_dict_keys = list(loss_dict.keys())
            
                train_verbose = "{:5d} / {:5d} ({:s}%):"
                verbose_args = [idx,loader_len,str(int(idx/loader_len*100)).zfill(3)]
                for key in loss_dict_keys:
                    train_verbose += "\t %s:{:.7f}"%key
                    verbose_args.append(loss_dict[key])
                
                # plot the cumulative mean total loss
                short_batch_df = pd.json_normalize(verbose_list) # this will be cumulative
                mean_total = short_batch_df.loc[:,'Total'].mean()
                train_verbose += "\t %s:{:.7f}"%"Mean_Total"
                verbose_args.append(mean_total)
                
                train_verbose = train_verbose.format(*verbose_args)                         
            
                logger.info(train_verbose)
        if popen.cuda_id != torch.device('cpu'):
            with torch.cuda.device(popen.cuda_id):
                torch.cuda.empty_cache()

def validate(dataloader,model,popen,epoch):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.teacher_forcing = False    # turn off teacher_forcing
    
    # ====== set up empty =====
    verbose_list=[]
    Y_ls = []
    pred_ls = []
    metric_dict = {}
    # ======== evaluate =======
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(dataloader):
            X,Y = put_data_to_cuda(data,popen,require_grad=False)
            Y_ls.append(Y.cpu().numpy())
            out = model(X)
            pred_ls.append(out.cpu().numpy())
            loss_dict = model.compute_loss(out,X,Y,popen)
            loss = loss_dict['Total']
            acc_dict = model.compute_acc(out,X,Y,popen)
            loss_dict.update(acc_dict) 
            
            loss_dict = utils.clean_value_dict(loss_dict)  # convert possible torch to single item
            verbose_list.append(loss_dict)

            if popen.cuda_id != torch.device('cpu'):   
                with torch.cuda.device(popen.cuda_id):  
                    torch.cuda.empty_cache()
            
          # # average among batch
    
    # ======== verbose ========
    Y_ay = np.concatenate(Y_ls,axis=0).flatten()
    pred_ay = np.concatenate(pred_ls,axis=0).flatten()

    if popen.model_type == 'RL_clf':
        metric_dict["F1"] = f1_score(Y_ay, pred_ay>0.5, average='binary')
        metric_dict["acc"] = np.mean(Y_ay == pred_ay)
        metric_dict["AUROC"] = roc_auc_score(Y_ay, pred_ay)
    else:
        metric_dict["r2"] = r2_score(Y_ay, pred_ay)
        metric_dict["pr"] = pearsonr(Y_ay, pred_ay)[0]
        metric_dict["spr"] = spearmanr(Y_ay, pred_ay)[0]
    
    verbose_df = pd.json_normalize(verbose_list)
        
    val_verbose = ""
    verbose_args = []
    verbose_dict = {key:verbose_df[key].mean() for key in verbose_df.columns}
    verbose_dict.update(metric_dict)

    for key,values in verbose_dict.items():
        val_verbose += "\t %s:{:.7f}"%key
        verbose_args.append(values)

    val_verbose = val_verbose.format(*verbose_args)                         

    logger.info(val_verbose)
    
    # what avg acc return : mean of  RL_Acc , Recons_Acc, Motif_Acc
    acc_col = list(acc_dict.keys())
    Avg_acc = np.mean(verbose_df.loc[:,acc_col].mean(axis=0))  
    
    # return these to save current performance
    return (verbose_df['Total'].mean(),Avg_acc) if 'RL_loss' not in verbose_df.keys() else (verbose_df['RL_loss'].mean(),verbose_df['RL_Acc'].mean())

def iter_train(loader_dict, model, optimizer, popen, epoch, verbose=True):

    logger = logging.getLogger("VAE")
    # loader_len = len(dataloader)       # number of iteration
    all_len = [len(loader[0]) for loader in loader_dict.values()]
    max_len = np.max(all_len)
    n_task = len(popen.cycle_set)
    total_len = max_len*n_task
    
    all_train = {task : iter(loader[0]) for task,loader in loader_dict.items()}
    def try_next(all_train, task):
        try:
            data = next(all_train[task])
            return data
        except StopIteration:
            all_train[task] = iter(loader_dict[task][0])
            data = next(all_train[task])
            return data
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    verbose_list=[]
    for idx in range(total_len):
        task = popen.cycle_set[idx%n_task]
        model.task = task
        
        data = try_next(all_train, task)
        
        X,Y = put_data_to_cuda(data,popen,require_grad=True)
        optimizer.zero_grad()
        
        out = model(X)
        loss_dict = model.compute_loss(out,X,Y,popen)
        loss = loss_dict['Total']
        acc_dict = model.compute_acc(out,X,Y,popen)
        loss_dict.update(acc_dict) # adding the acc ditc into loss dict
         
        loss.backward()        
        optimizer.step()
        
        # ====== update lr =======
        if type(optimizer) == ScheduledOptim:
            lr = optimizer.update_learning_rate()      # see model.optim.py
            loss_dict["lr"]=lr
        elif popen.optimizer == 'Adam':
            lr = optimizer.param_groups[0]['lr']
        if popen.loss_schema != 'constant':
            popen.chimerla_weight = popen.loss_schedualer._update(loss_dict)
            for t in popen.tasks:
                loss_dict[popen.loss_schema+"_wt_"+t] = popen.chimerla_weight[t]
                
        with torch.no_grad():   
            loss_dict= utils.clean_value_dict(loss_dict)
            verbose_list.append(loss_dict)
       
        # ======== verbose ========
        # record result 5 times for a epoch
        if verbose:
            if idx % int(total_len/5) == 0:
                
                # plot that in loss dict
                loss_dict_keys = list(loss_dict.keys())
            
                train_verbose = "{:5d} / {:5d} ({:s}%):"
                verbose_args = [idx,total_len,str(int(idx/total_len*100)).zfill(3)]
                for key in loss_dict_keys:
                    train_verbose += "\t %s:{:.7f}"%key
                    verbose_args.append(loss_dict[key])
                
                # plot the cumulative mean total loss
                short_batch_df = pd.json_normalize(verbose_list) # this will be cumulative
                mean_total = short_batch_df.loc[:,'Total'].mean()
                train_verbose += "\t %s:{:.7f}"%"Mean_Total"
                verbose_args.append(mean_total)
                
                train_verbose = train_verbose.format(*verbose_args)                         
            
                logger.info(train_verbose)

        if popen.cuda_id != torch.device('cpu'):
            with torch.cuda.device(popen.cuda_id):
                torch.cuda.empty_cache()

def plot_TR(Y_ay, pred_ay, subset, which_set, ckpt_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    fig, ax = plt.subplots()
    ax.scatter(Y_ay, pred_ay, alpha=0.5)
    min_ = np.floor(min(Y_ay.min(), pred_ay.min()))
    max_ = np.ceil(max(Y_ay.max(), pred_ay.max()))   
    # ax.plot([Y_ay.min(), Y_ay.max()], [Y_ay.min(), Y_ay.max()], 'k--', lw=4)
    ax.set_xticks(np.arange(min_, max_ + 1.1, 1))
    ax.set_yticks(np.arange(min_, max_ + 1.1, 1))
    ax.set_xlabel('True Log TE')
    ax.set_ylabel('Predicted Log TE')
    os.makedirs("figs", exist_ok=True)
    plt.show()
    plt.savefig(f'figs/TE_{ckpt_path}_{subset}_{which_set}.png')

def cycle_validate(loader_dict, model, optimizer, popen, epoch , which_set=1, return_=False, ckpt_path=""):

    logger = logging.getLogger("VAE")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # ====== set up empty =====
    # model.loss_dict_keys = ['RL_loss', 'Recons_loss', 'Motif_loss', 'Total', 'RL_Acc', 'Recons_Acc', 'Motif_Acc']
    verbose_list=[]
    r2_dict = {}
    Y_n_pred = {}
    # ======== evaluate =======
    
    
    for subset, dataloader in loader_dict.items():
        
        # fix 
        # model.task = subset #[yyk] remove
        # # logger.info("        =======================|     fix      |=======================        ")
        # model = utils.fix_parameter(model, popen.modual_to_fix[0])
        # train(dataloader[0], model, optimizer, popen, epoch, verbose=True)
        Y_ls = []
        pred_ls = []
        with torch.no_grad():
            model.eval()
            for idx,data in enumerate(dataloader[which_set]):
                X,Y = put_data_to_cuda(data,popen,require_grad=False)
                out = model(X)

                Y_ls.append(Y.detach().cpu().numpy())
                pred_ls.append(out.detach().cpu().numpy())

                loss_dict = model.compute_loss(out,X,Y,popen)
                loss_dict['%s_loss'%subset] = loss_dict['Total']
                acc_dict = model.compute_acc(out,X,Y,popen)
                acc_dict['%s_Acc'%subset] = acc_dict['Acc']
                loss_dict.update(acc_dict) 
                
                loss_dict = utils.clean_value_dict(loss_dict)  # convert possible torch to single item
                verbose_list.append(loss_dict)
                
                if popen.cuda_id != torch.device('cpu'):
                    with torch.cuda.device(popen.cuda_id):  
                        torch.cuda.empty_cache()
        
        Y_ay = np.concatenate(Y_ls,axis=0).flatten()
        pred_ay = np.concatenate(pred_ls,axis=0).flatten()
        if return_:
            Y_n_pred[subset] = (Y_ay, pred_ay)
        r2_dict[f"{subset}_r2"] = r2_score(Y_ay, pred_ay)
        r2_dict[f"{subset}_pr"] = pearsonr(Y_ay, pred_ay)[0]
        r2_dict[f"{subset}_spr"] = spearmanr(Y_ay, pred_ay)[0]

        plot_TR(Y_ay, pred_ay, subset, which_set, ckpt_path)
        

          # # average among batch
    
    # ======== verbose ========
    
    verbose_df = pd.json_normalize(verbose_list)
        
    val_verbose = ""
    verbose_args = []
    verbose_dict = {key:verbose_df[key].mean() for key in verbose_df.columns}
    verbose_dict.update(r2_dict)
    for key,values in verbose_dict.items():
        val_verbose += "\t %s:{:.7f}"%key
        verbose_args.append(values)
        
    val_verbose = val_verbose.format(*verbose_args)                         

    logger.info(val_verbose)
    
    # what avg acc return : mean of  RL_Acc , Recons_Acc, Motif_Acc
    acc_col = list(acc_dict.keys())
    Avg_acc = np.mean(verbose_df.loc[:,acc_col].mean(axis=0))  
    
    if return_:
        return verbose_dict, Y_n_pred
    else:
        return verbose_dict

         
def put_data_to_cuda(data,popen,require_grad=True):

    X,y = data       
    device = popen.cuda_id 
    # for X is a list [seq, uAUG ....]
    if popen.other_input_columns is not None:
        X = [x_i.float().to(device) for x_i in X]
        if require_grad:
            for x_i in X:
                x_i.required_grad = True

    # X is not a list : seq
    else:
        X = X.float().to(device)
        if require_grad:
            X.required_grad = True  # check !!!
    
    Y = y.float().to(device)
    
    # Y = Y if X.shape == Y.shape else None  # for mask data
    return X,Y