import os
import sys
import torch
from torch import nn
import numpy as np

class self_attention(nn.Module):
    
    def __init__(self,in_channel,n_head,d_k,d_v): 
        super(self_attention,self).__init__()
        self.n_head=n_head
        self.d_k = d_k
        self.d_v = d_v
        self.W_dict = nn.ModuleDict({"Wq" : nn.Linear(in_channel,n_head*d_k),
                                     "Wk" : nn.Linear(in_channel,n_head*d_k),
                                     "Wv" : nn.Linear(in_channel,n_head*d_v)})

    def forward(self,X):
        
        # some dimension 
        dk_sqrt = int(np.sqrt(self.d_k))
        
        querys = self.W_dict['Wq'](X)     # B*X_dim*dk
        keys = self.W_dict['Wk'](X)       # B* hs_out -> B*64
        values = self.W_dict['Wv'](X)    # B* hs_out -> B*128
        
        sim_M = torch.bmm(querys,keys.transpose(1,2))/8  # B* X_dim * X_dim
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.bmm(attention,values).squeeze(2)    # B*X_dim*1 -> B*X_dim
        
        return result
    
class task_attention(nn.Module):
    """
    Task Attention Layer, it contains a task specific mask `Wq` as the Query. Key is computed from the input 
    
    Arguments:

        d_v : int , dimension of the network width input , i.e : (Batch_size, d_v)
        d_k : int , dimension of the Query and Key.
    """
    def __init__(self,d_v:int,d_k=64): 
        super(task_attention,self).__init__()
    
        self.d_k = d_k
        self.d_v = d_v
        self.W_dict = nn.ModuleDict({"Wq" : nn.Linear(d_k,d_v),
                                     "Wk" : nn.Linear(d_v,d_k)})

    def forward(self,X):
        # i.e X : (B ,128)
        # if len(X.shape) == 2:
        #     X = X
        
        # some dimension 
        dk_sqrt = int(np.sqrt(self.d_k))
        
        keys = self.W_dict['Wk'](X).squeeze(2)       # (B, 128) -> (B ,64)
        
        # dot product similarity is used here , which is implemented by `nn.Linear`
        sim_M =self.W_dict['Wq'](keys) /dk_sqrt  # (B,128,64) * (B,64,128)  ->  (B, 128, 128)
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.mul(attention,X)   # (B, 128)  * (B, 128) -> B*128
        
        return result

class task_attention(nn.Module):
    """
    Task Attention Layer, it contains a task specific mask `Wq` as the Query. Key is computed from the input 
    
    Arguments:

        d_v : int , dimension of the network width input , i.e : (Batch_size, d_v)
        d_k : int , dimension of the Query and Key.
    """
    def __init__(self,d_v:int,d_k=64): 
        super(task_attention,self).__init__()
    
        self.d_k = d_k
        self.d_v = d_v
        self.W_dict = nn.ModuleDict({"Wq" : nn.Linear(d_v,d_k),
                                     "Wk" : nn.Linear(d_v,d_k)})

    def forward(self,X):
        # i.e X : (B ,128)
        # if len(X.shape) == 2:
        #     X = X
        
        # some dimension 
        dk_sqrt = int(np.sqrt(self.d_k))
        query = self.W_dict['Wq'](X).squeeze(2)
        keys = self.W_dict['Wk'](X).squeeze(2)       # (B, 128) -> (B ,64)
        
        # dot product similarity is used here , which is implemented by `nn.Linear`
        sim_M =self.W_dict['Wq'](keys) /dk_sqrt  # (B,128,64) * (B,64,128)  ->  (B, 128, 128)
        attention = torch.softmax(sim_M,dim=-1)
        # result
        result = torch.mul(attention,X)   # (B, 128)  * (B, 128) -> B*128
        
        return result