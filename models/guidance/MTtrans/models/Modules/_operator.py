import torch 
import math
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout

class Conv1d_block(nn.Module):
    """
    the Convolution backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride, padding_ls=None,diliation_ls=None,pad_to=None, activation='ReLU'):
        """
        Argument
            channel_ls : list, [int] , channel for each conv layer
            kernel_size : int
            stride :  list , [int]
            padding_ls :   list , [int]
            diliation_ls : list , [int]
        """
        super(Conv1d_block,self).__init__()
        ### property
        self.activation = activation
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        if padding_ls is None:
            self.padding_ls = [0] * (len(channel_ls) - 1)
        else:
            assert len(padding_ls) == len(channel_ls) - 1
            self.padding_ls = padding_ls
        if diliation_ls is None:
            self.diliation_ls = [1] * (len(channel_ls) - 1)
        else:
            assert len(diliation_ls) == len(channel_ls) - 1
            self.diliation_ls = diliation_ls
        
        self.encoder = nn.ModuleList(
            #                   in_C         out_C           padding            diliation
            [self.Conv_block(channel_ls[i],channel_ls[i+1],self.padding_ls[i],self.diliation_ls[i],self.stride[i]) for i in range(len(self.padding_ls))]
        )
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        
        activation_layer = eval(f"nn.{self.activation}")
        
        block = nn.Sequential(
                nn.Conv1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation),
                nn.BatchNorm1d(out_Chan),
                activation_layer())
        
        return block
    
    def forward(self,x):
        if x.shape[2] == 4:
            out = x.transpose(1,2)
        else:
            out = x
        for block in self.encoder:
            out = block(out)
        return out
    
    def forward_stage(self,x,stage):
        """
        return the activation of each stage for exchanging information
        """
        assert stage < len(self.encoder)
        
        out = self.encoder[stage](x)
        return out

    def cal_out_shape(self,L_in=100,padding=0,diliation=1,stride=2):
        """
        For convolution 1D encoding , compute the final length 
        """
        L_out = 1+ (L_in + 2*padding -diliation*(self.kernel_size-1) -1)/stride
        return L_out
    
    def last_out_len(self,L_in=100):
        for i in range(len(self.padding_ls)):
            padding = self.padding_ls[i]
            diliation = self.diliation_ls[i]
            stride = self.stride[i]
            L_in = self.cal_out_shape(L_in,padding,diliation,stride)
        # assert int(L_in) == L_in , "convolution out shape is not int"
        
        return int(L_in) if L_in >=0  else 1
    
class ConvTranspose1d_block(Conv1d_block):
    """
    the Convolution transpose backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride,padding_ls=None,diliation_ls=None,pad_to=None):
        channel_ls = channel_ls[::-1]
        stride = stride[::-1]
        padding_ls =  padding_ls[::-1] if padding_ls  is not None else  [0] * (len(channel_ls) - 1)
        diliation_ls =  diliation_ls[::-1] if diliation_ls  is not None else  [1] * (len(channel_ls) - 1)
        super(ConvTranspose1d_block,self).__init__(channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to)
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        """
        replace `Conv1d` with `ConvTranspose1d`
        """
        block = nn.Sequential(
                nn.ConvTranspose1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation=dilation),
                nn.BatchNorm1d(out_Chan),
                nn.ReLU())
        
        return block
    
    def cal_out_shape(self,L_in,padding=0,diliation=1,stride=1,out_padding=0):
        #                  L_in=100,padding=0,diliation=1,stride=2
        """
        For convolution Transpose 1D decoding , compute the final length
        """
        L_out = (L_in -1 )*stride + diliation*(self.kernel_size -1 )+1-2*padding + out_padding 
        return L_out


class linear_block(nn.Module):
    def __init__(self,in_Chan,out_Chan,dropout_rate=0.2):
        """
        building block func to define dose network
        """
        super(linear_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_Chan,out_Chan),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_Chan),
            nn.ReLU()
        )
    def forward(self,x):
        return self.block(x)


class Self_Attention(nn.Module):
    """
    self attention operator for Conv1d sequences output
    """
    def __init__(self, in_dim:int, out_dim:int, qk_dim:int, n_head:int):
        super().__init__()

        self.n_head = n_head
        self.total_qk_dim = qk_dim * n_head
        self.transform = nn.ModuleDict({
            k : nn.Linear(in_dim, self.total_qk_dim) for k in ['k', 'q', 'v']
        })

        self.fc_out = nn.Linear(self.total_qk_dim, out_dim)

    def dim_rerrange(self, x):

        # first break down total qk dimension
        # then transpose length with heads
        x1 = rearrange(x, "b l (n c) -> b n c l", n=self.n_head) 
        return x1
    
    def _get_attention_map(self,X):
        """
        break the forward function to access attention mat
        """
        # assume we have a 3 dimension input X (b, len, in_dim)
        # each out in qkv is also 3 dimension  (b, len , qk_dim)
        qkv = [self.transform[key](X) for key in ['k', 'q', 'v']]
        q, k, v = map(self.dim_rerrange, qkv)

        # here i and j is the channel
        sim = torch.einsum("b n c i, b n c j -> b n i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() 
        attn = sim.softmax(dim=-1)
        return attn, v

    def forward(self, X):
    
        attn, v = self._get_attention_map(X)

        out = torch.einsum("b n i j, b n c j -> b n i c", attn, v)
        out = rearrange(out, "b n i c -> b i (n c)")
        return self.fc_out(out)


class Self_Attention_for_GP(Self_Attention):
    """
    self attention operator for Conv1d sequences Global Pooling output
    The input has 2 dimension (no length dim), 
    """
    def __init__(self, in_dim:int, out_dim:int, qk_dim:int, n_head:int):
        super().__init__(in_dim, out_dim, qk_dim, n_head)

    def _get_attention_map(self,X):
        # assume we have a 3 dimension input X (b, len, in_dim)
        # each out in qkv is also 3 dimension  (b, len , qk_dim)
        qkv = [self.transform[key](X) for key in ['k', 'q', 'v']]
        q, k, v = map(
            lambda x : rearrange(x, "b (n c)-> b n c"), qkv
        )

        # here i and j is the channel
        sim = torch.einsum("b n i, b n j -> b n i j", q, k).softmax(dim=-2, keepdim=True)
        sim = sim - attn.amax(dim=-1, keepdim=True).detach() 
        attn = sim.softmax(dim=-1)
        return attn, v

    def forward(self, X):
        # 
        attn, v = self._get_attention_map(X)

        out = torch.einsum("b n i j, b n j -> b n i", attn, v)
        out = rearrange(out, "b n i -> b (n i)")
        return self.fc_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
        
    def forward(self, x):
        x = self.norm(x.transpose(1,2))
        return self.fn(x.transpose(1,2))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2 
        embeddings = math.log(10000) / (half_dim -1) # why do we minus 1 ?
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -embeddings)
        embeddings = time[:, None] * embeddings[None, :] # expand to 2 dimension
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings