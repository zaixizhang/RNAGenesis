import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import utils
import json
from models import Modules
import configparser
import logging

class Auto_popen(object):
    def __init__(self,config_file):
        """
        read the config_fiel
        """
        # machine config path
        self.shuffle = True
        self.script_dir = utils.script_dir
        self.data_dir = utils.data_dir
        self.log_dir = utils.log_dir
        self.pth_dir = utils.pth_dir
        self.set_attr_as_none(['te_net_l2','loss_fn','modual_to_fix','other_input_columns','pretrain_pth','kfold_index'])
        self.split_like = False
        self.loss_schema = 'constant'
        
        # transform to dict and convert to  specific data type
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file
        self.config_dict = {item[0]: eval(item[1]) for item in self.config.items('DEFAULT')}
        
        # assign some attr from config_dict         
        self.set_attr_from_dict(self.config_dict.keys())
        self.check_run_and_setting_name()                          # check run name
        self._dataset = "_" + self.dataset if self.dataset != '' else self.dataset
        # the saving direction
        self.path_category = self.config_file.split('/')[-4]
        self.vae_log_path = config_file.replace('.ini','.log')
        

        self.Resumable = False

        # covariates for other input
        self.n_covar = len(self.other_input_columns) if self.other_input_columns is not None else 0
        
        # generate self.model_args
        self.get_model_config()
        if self.model_type in ["RL_kel"]:
            self.other_input_columns+=['freq_AAA', 'freq_AAG', 'freq_AAC', 'freq_AAT', 'freq_AGA', 'freq_AGG', 
                'freq_AGC', 'freq_AGT', 'freq_ACA', 'freq_ACG', 'freq_ACC', 'freq_ACT', 'freq_ATA', 'freq_ATG', 
                    'freq_ATC', 'freq_ATT', 'freq_GAA', 'freq_GAG', 'freq_GAC', 'freq_GAT', 'freq_GGA', 'freq_GGG', 
                    'freq_GGC', 'freq_GGT', 'freq_GCA', 'freq_GCG', 'freq_GCC', 'freq_GCT', 'freq_GTA', 'freq_GTG', 
                    'freq_GTC', 'freq_GTT', 'freq_CAA', 'freq_CAG', 'freq_CAC', 'freq_CAT', 'freq_CGA', 'freq_CGG', 
                    'freq_CGC', 'freq_CGT', 'freq_CCA', 'freq_CCG', 'freq_CCC', 'freq_CCT', 'freq_CTA', 'freq_CTG', 
                    'freq_CTC', 'freq_CTT', 'freq_TAC', 'freq_TAT', 'freq_TGG', 'freq_TGC', 'freq_TGT', 'freq_TCA', 
                    'freq_TCG', 'freq_TCC', 'freq_TCT', 'freq_TTA', 'freq_TTG', 'freq_TTC', 'freq_TTT']


    @property
    def vae_pth_path(self):
        save_to = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        if self.kfold_index is None:
            pth = os.path.join(save_to, self.run_name + '-model_best.pth')
        elif type(self.kfold_index) == int:
            k = self.kfold_index
            pth = os.path.join(save_to, self.run_name + f'-model_best_cv{k}.pth')
        return pth
    
    def set_attr_from_dict(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,self.config_dict[attr])
    
    def set_attr_as_none(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,None)

    def check_run_and_setting_name(self):
        file_name = self.config_file.split("/")[-1]
        dir_name = self.config_file.split("/")[-2]
        self.setting_name = dir_name
        assert self.run_name == file_name.split(".")[0]
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        
        if self.model_type in dir(Modules):
            self.Model_Class = eval("Modules.{}".format(self.model_type))
        else:
            raise NameError("not such model type")
        
        # conv_args define the soft-sharing part
        conv_args = ["channel_ls","kernel_size","stride","padding_ls","diliation_ls","pad_to"]
        self.conv_args = tuple([self.__getattribute__(arg) for arg in conv_args])
        
        # left args dfine the tower part in which the arguments are different among tasks
        left_args={# Backbone models
                    'RL_regressor':["tower_width","dropout_rate"],
                    'RL_clf':["n_class","tower_width","dropout_rate"],
                    'RL_gru':["tower_width","dropout_rate"],
                    'RL_FACS': ["tower_width","dropout_rate"],
                    'RL_hard_share':["tower_width","dropout_rate", "activation","cycle_set" ],
                    'RL_covar_reg':["tower_width","dropout_rate", "activation", "n_covar", "cycle_set" ],
                    'RL_covar_intercept':["tower_width","dropout_rate", "activation", "n_covar", "cycle_set" ],
                    "RL_kel":["tower_width","dropout_rate", "activation", "n_covar", "cycle_set" ],
                    'RL_mish_gru':["tower_width","dropout_rate"],
                    # GP models
                    'GP_net': ['tower_width', 'dropout_rate', 'global_pooling', 'activation', 'cycle_set'],
                    'Frame_GP': ['tower_width', 'dropout_rate', 'activation', 'cycle_set'],
                    'RL_Atten': ['qk_dim', 'n_head', 'n_atten_layer', 'tower_width', 'dropout_rate', 'activation', 'cycle_set'],
                    # Koo net
                    'Conf_CNN' : ['pool_size'],
                    }[self.model_type]
        
        self.model_args = [self.conv_args] + [self.__getattribute__(arg) for arg in left_args]
        
            
    def check_experiment(self,logger):
        """
        check any unfinished experiment ?
        """
        log_save_dir = os.path.dirname(self.vae_log_path)
        pth_save_dir = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        # make dirs 
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        if not os.path.exists(pth_save_dir):
            os.makedirs(pth_save_dir)
        
        # check resume
        if os.path.exists(self.vae_log_path) & os.path.exists(self.vae_pth_path):
            self.Resumable = True
            logger.info(' \t \t ==============<<<  Experiment detected  >>>============== \t \t \n')
            
    def update_ini_file(self,E,logger):
        """
        E  is the dict contain the things to update
        """
        # update the ini file
        self.config_dict.update(E)
        strconfig = {K: repr(V) for K,V in self.config_dict.items()}
        self.config['DEFAULT'] = strconfig
        
        with open(self.config_file,'w') as f:
            self.config.write(f)
        
        logger.info('   ini file updated    ')
        
    def chimera_weight_update(self):
        # TODO : progressively update the loss weight between tasks
        
        # TODO : 1. scale the loss into the same magnitude
        
        # TODO : 2. update the weight by their own learning progress
        
        return None