import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout
from collections import OrderedDict
from ._operator import linear_block, Self_Attention, Residual, PreNorm, SinusoidalPositionEmbeddings
from .Backbone import RL_regressor, RL_hard_share



class GP_net(RL_hard_share):
    """
    A very special RL regressor model which spacial output of the last conv1d is collasped.
    """
    def __init__(self, 
                 conv_args,
                 tower_width :int = 512,
                 dropout_rate : int = 0.3,
                 global_pooling:str='max',
                 activation:str='Mish',
                 tasks =['unmod1']
                 ):
        super().__init__(conv_args, tower_width, dropout_rate, activation, tasks)

        self.dropout_rate = dropout_rate
        # ------- Global pooling -------
        self.pool_fn = nn.MaxPool1d(self.out_length) if global_pooling=='max' else nn.AvgPool1d(self.out_length)
        
        # ------- linear block -------
        tower_block = lambda c,w : nn.Sequential(
                                                linear_block(c, w, dropout_rate=self.dropout_rate),
                                                nn.Linear(w,1))
        
        
        self.tower = nn.ModuleDict({task: tower_block(self.channel_ls[-1], tower_width) for task in self.all_tasks})
    
    def forward_Global_pool(self, Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = self.pool_fn(Z) 

        # pool and 
        if len(Z_flat.shape) == 3:
            Z_flat = Z_flat.view(batch_size, self.channel_ls[-1])
        return Z_flat

    
    def forward(self, X):
        
        task = self.task # pass in cycle_train.py
        # Con block
        Z = self.soft_share(X)
        # pool
        Z_flat = Z.amax(dim=-1)
        # tower
        out = self.tower[task](Z_flat)
        return out

class Frame_GP(RL_hard_share):
    """
    A very special RL regressor model which spacial output of the last conv1d is collasped.
    """
    def __init__(self, 
                 conv_args,
                 tower_width :int = 512,
                 dropout_rate : int = 0.3,
                 activation:str='Mish',
                 tasks =['unmod1']
                 ):
        super().__init__(conv_args, tower_width, dropout_rate, activation, tasks)

        self.dropout_rate = dropout_rate
        # ------- Global pooling -------
        tower_block = lambda c,w : nn.Sequential(
                                                linear_block(c, w, dropout_rate=self.dropout_rate),
                                                nn.Linear(w,1))
        
        self.tower = nn.ModuleDict({task: tower_block(3*self.channel_ls[-1], tower_width) for task in self.all_tasks})
    
    def forward(self, X):
        task = self.task # pass in cycle_train.py
        # Con block  
        Z = self.soft_share(X) # B Channel Length

        length = Z.shape[2]
        
        frame1 = np.arange(0, length, 3).tolist()
        frame2 = np.arange(1, length, 3).tolist()
        frame3 = np.arange(2, length, 3).tolist()

        Z1 = Z[:,:, frame1].amax(dim=-1)
        Z2 = Z[:,:, frame2].amax(dim=-1)
        Z3 = Z[:,:, frame3].amax(dim=-1)

        frame_Z = torch.cat([Z1, Z2, Z3], dim=1)

        return self.tower[task](frame_Z)

class RL_Atten(RL_hard_share):
    """
    repalce the final fc layers to self-atten, this allows for motif interaction 
    """
    def __init__(self, 
                 conv_args,
                 qk_dim:int = 64, 
                 n_head:int = 8,
                 n_atten_layer:int = 3,
                 tower_width :int = 512,
                 dropout_rate : int = 0.3,
                 activation:str='Mish',
                 tasks =['MPA_U']):

        super().__init__(conv_args, tower_width, dropout_rate, activation, tasks)

        self.position_emb_dim = 32
        self.n_atten_layer = n_atten_layer
        self.qk_dim = qk_dim
        self.n_head = n_head
        
        self.position_emb = SinusoidalPositionEmbeddings(dim=32)
        attn_channel = self.channel_ls[-1]
        self.tower = nn.ModuleDict({task: self.tower_block(attn_channel, tower_width, n_atten_layer, 32) for task in self.all_tasks})

    def tower_block(self, attn_ch, tower_width, n_layer):
        Atten_block = []
        for i in range(n_layer):
            # input and output are the same dimension
            Atten_block += [
                (f"Res_Attn_{i+1}" , Self_Attention(attn_ch+self.position_emb_dim, attn_ch, self.qk_dim, self.n_head)),
                (f"Attn_act_{i+1}" , nn.SiLU())
            ]
        
        # output layer
        output = [
            (f"pool_layer", nn.Linear(self.out_dim, tower_width)),
            (f"pool_act", nn.SiLU()),
            (f"out_layer", nn.Linear(tower_width, 1))
        ]

        tower = nn.ModuleDict(
            {"Attention":nn.Sequential(OrderedDict(Atten_block)),
             "output_layer":nn.Sequential(OrderedDict(output)),
            }
        )
        return tower


    @torch.no_grad()
    def _get_attention_map(self, X, task):
        """
        a function to quickly access attention matrix from input
        """
        Z = self.soft_share(X).transpose(1,2)
        # this func has 2 return
        attn, _ = self.tower[task].Res_Attn_1._get_attention_map(Z) 
        return attn

    def forward(self, X):
        
        task = self.task # pass in cycle_train.py
        batch_size = X.shape[0]
        # Con block
        Z = self.soft_share(X).transpose(1,2)
        
        # rearrange Z to batch length channel
        Attn_out = self.tower[task]['Attention'](Z).view(batch_size, -1)
        out = self.tower[task]['output_layer'](Attn_out)
        return out



