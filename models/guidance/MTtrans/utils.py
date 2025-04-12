from sklearn.preprocessing import OneHotEncoder
import logging
import numpy as np
import pandas as pd
import os
import json
import re
import torch
import collections
# from models import reader
from models.ScheduleOptimizer import ScheduledOptim

print(os.path.dirname(__file__))

# ====================|   some path   |=======================
global script_dir
global data_dir
global log_dir
global pth_dir
# global cell_lines

global egfp_seq

with open(os.path.join(os.path.dirname(__file__),"machine_configure.json"),'r') as f:
    config = json.load(f)   

script_dir = config['script_dir']
data_dir = config['data_dir']
log_dir = config['log_dir']
pth_dir = config['pth_dir']



# =====================| one hot encode |=======================

class Seq_one_hot(object):
    def __init__(self,seq_type='nn',seq_len=100):
        """
        initiate the sequence one hot encoder
        """
        self.seq_len=seq_len
        self.seq_type =seq_type
        self.enable_encoder()
        
    def enable_encoder(self):
        if self.seq_type == 'nn':
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.drop_idx_ = None
            self.encoder.categories_ = [np.array(['A', 'C', 'G', 'T'], dtype='<U1')]*self.seq_len

    def discretize_seq(self,data):
        """
        discretize sequence into character
        argument:
        ...data: can be dataframe with UTR columns , or can be single string
        """
        if type(data) is pd.DataFrame:
            return np.stack(data.UTR.apply(lambda x: list(x)))
        elif type(data) is str:
            return np.array(list(data))
    
    def transform(self,data,flattern=True):
        """
        One hot encode
        argument:
        data : is a 2D array
        flattern : True
        """
        X = self.encoder.transform(data)                             # 400 for each seq
        X_M = np.stack([seq.reshape(self.seq_len,4) for seq in X])   # i.e 100*4
        return X if flattern else X_M
    
    def d_transform(self,data,flattern=True):
        """
        discretize data and put into transform
        """
        X = self.discretize_seq(data)
        return self.transform(X,flattern)


# =====================|   logger       |=======================

def setup_logs(vae_log_path,level=None):
    """

    :param save_dir:  the directory to set up logs
    :param type:  'model' for saving logs in 'logs/cpc'; 'imp' for saving logs in 'logs/imp'
    :param run_name:
    :return:logger
    """
    # initialize logger
    logger = logging.getLogger("VAE")
    logger.setLevel(logging.INFO)
    if level=='warning':
        logger.setLevel(logging.WARNING)

    # create the logging file handler
    log_file = os.path.join(vae_log_path)
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def clean_value_dict(dict):
    """
    deal with verbose dict where the values maybe torch object, extact the item and return clean dict
    """
    clean_dict={}
    for k,v in dict.items():
        
        try:
            v = v.item()
        except:
            v = v
        clean_dict[k] = v
    return clean_dict

def fix_parameter(model,modual_to_fix,fix_or_unfix=False):
    """
    for a given model, fix part of the parameter to fine-tuning / transfering 
    args:
    model : `nn.Modual`,initiated model instance
    modual_to_fix : str, define which part of the model will not update by gradient 
                    e.g. "soft_share" then 
    """
    
    fix_part = eval("model."+modual_to_fix)   # e.g. model.shoft_share
     
    for param in fix_part.parameters():
            param.requires_grad = fix_or_unfix
    
    return model

def unfix_parameter(model,modual_to_fix,fix_or_unfix=False):
    return fix_parameter(model,modual_to_fix,fix_or_unfix=True)

def snapshot(vae_pth_path, state):
    logger = logging.getLogger("VAE")
    # torch.save can save any object
    # dict type object in our cases
    torch.save(state, vae_pth_path)
    logger.info("Snapshot saved to {}\n".format(vae_pth_path))


def load_model(popen,model,logger=None):
    
    info = lambda x: print(x) if logger==None else logger.info(x)
    
    checkpoint = torch.load(popen.vae_pth_path, map_location=torch.device('cpu')) 
    if isinstance(checkpoint['state_dict'], collections.OrderedDict):
            # optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint['state_dict']
    
    info(' \t \t ==============<<< encoder load from >>>============== \t \t ')
    info(" \t"+popen.vae_pth_path)
    
    return model
    
def get_config_cuda(config_file):
    with open(config_file,'r') as f:
        lines = f.read_lines()
        for line in lines:
            if "cuda_id =" in line:
                device = line.split("=")[1].strip()
                break
    device = int(device) if device.isdigit() else device
    return device

def resume(popen,optimizer,logger):
    """
    for a experiment, check whether it;s a new run, and create dir 
    """
    #run_name = model_stype + time.strftime("__%Y_%m_%d_%H:%M"))
    
    if popen.Resumable:
        
        checkpoint = torch.load(popen.vae_pth_path, map_location=torch.device('cpu'))   # xx-model-best.pth
        previous_epoch = checkpoint['epoch']
        previous_loss = checkpoint['validation_loss']
        previous_acc = checkpoint['validation_acc']
        
        
        # very important
        if (type(optimizer) == ScheduledOptim):
            optimizer.n_current_steps = popen.n_current_steps
            optimizer.delta = popen.delta
        
        logger.info(" \t \t ========================================================= \t \t ")
        logger.info(' \t \t ==============<<< Resume from checkpoint>>>============== \t \t \n')
        logger.info(" \t"+popen.vae_pth_path+'\n')
        logger.info(" \t \t ========================================================= \t \t \n")
        
        return previous_epoch,previous_loss,previous_acc
        
    
egfp_seq = "atgggcgaattaagtaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttct"
eGFP_seq = egfp_seq.upper()