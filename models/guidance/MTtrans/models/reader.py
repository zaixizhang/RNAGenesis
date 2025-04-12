import os
import sys
import copy
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import torch
import numpy as np
import pandas as pd
from torch import nn

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from torch.utils.data import  DataLoader, Dataset ,random_split,IterableDataset
from sklearn.model_selection import KFold,train_test_split
from bucket_sampler import Bucket_Sampler
from utils import Seq_one_hot

global script_dir
global data_dir

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"machine_configure.json"),'r') as f:
    config = json.load(f)

script_dir = config['script_dir']
data_dir = config['data_dir']

def one_hot(seq,complementary=False):
    """
    one_hot encoding on sequence
    complementary: encode nucleatide into complementary one
    """
    # setting
    seq = list(seq.replace("U","T"))
    seq_len = len(seq)
    complementary = -1 if complementary else 1
    # compose dict
    keys = ['A', 'C', 'G', 'T'][::complementary]
    oh_dict = {keys[i]:i for i in range(4)}
    # array
    oh_array = np.zeros((seq_len,4))
    for i,C in enumerate(seq):
        try:
            oh_array[i,oh_dict[C]]=1
        except:
            continue      # for nucleotide that are not in A C G T   
    return oh_array 

def PWM_score(seq, pwm, pad_to=None, stride=1, agg_func=np.max):
    """
    seq : the target sequence, [str | ndarray], where to look for the occurance of motifs 
    pwm : position weight matrix of shape n, 4, **['A','C','G','T']
    agg_func : callable, convert the scores of all windows into single value, or 
    """
    # tidy check mat shape
    if type(seq) == str:
        one_hot_x = one_hot(seq)
    elif isinstance(seq, np.ndarray):
        one_hot_x = seq
    assert len(one_hot_x.shape) == 2
    assert len(pwm.shape) == 2
    if one_hot_x.shape[1] != 4:
        one_hot_x = one_hot_x.T
    if pwm.shape[1] != 4:
        pwm = pwm.T
    if pad_to is not None:
        one_hot_x = pad_zeros(one_hot_x, pad_to)
    
    # 
    len_pwm = pwm.shape[0]
    len_seq = one_hot_x.shape[0]
    Seq_I = []
    for i in range(0, len_seq-len_pwm, stride):
        # stride = 1
        seqlet_i = one_hot_x[i:i+len_pwm]
        if seqlet_i.shape[0] == len_pwm:
            Seq_I.append(seqlet_i.flatten())
    Seq_I = np.stack(Seq_I)
    score_I =  Seq_I @ pwm.values.reshape(-1,1)
    return agg_func(score_I) if agg_func is not None else score_I

def read_bpseq(test_bpseq_path):
    """
    read bpseq file, extract sequence and ptable
    """
    with open(test_bpseq_path,'r') as f:
        test_bpseq = f.readlines()
        f.close()

    for i in range(8):
        if test_bpseq[i].startswith('1'):
            start_index = i 

    bp_seq = test_bpseq[start_index:]

    seq = ''.join([line.strip().split(" ")[1] for line in bp_seq])
    ptable = [int(line.strip().split(" ")[2]) for line in bp_seq]

    assert len(ptable) == int(bp_seq[-1].split(" ")[0])

    return seq,ptable

def pad_zeros(X,pad_to):
    """
    zero padding at the right end of the sequence
    """
    if pad_to == 0:
        # pad_to = 8*(X.shape[0]//8 + 1) + 1
        try:
            seq_len = X.shape[0]
        except:
            seq_len = X[0].shape[0]
        pad_to = 3*(seq_len//3 + 1) + 1
        
    seq_len = X.shape[0] if isinstance(X, np.ndarray) else X[0].shape[0]
    gap = pad_to - seq_len 
    
    #  here we change to padding ahead  , previously  nn.ZeroPad2d([0,0,0,gap])
    pad_fn = nn.ZeroPad2d([0,0,gap,0])  #  (padding_left , padding_right , padding_top , padding_bottom )
    # gap_array = np.zeros()
    if isinstance(X, list):          # additional input
        X_padded = [pad_fn(torch.tensor(X[0])), X[1]]
    elif isinstance(X, np.ndarray):
        X_padded = pad_fn(torch.tensor(X))
    return X_padded

def pack_seq(ds_zls:list,pad_to:int):
    X_ts = [X for X,Y in ds_zls]
    if pad_to == 0:
        try:
            max_len = np.max([X.shape[0] for X in X_ts])
        except:
            max_len = np.max([X[0].shape[0] for X in X_ts])
        
        pad_to = 3*(max_len//3 + 1) + 1
    
    if isinstance(X_ts[0],np.ndarray):
        X_packed = torch.stack([pad_zeros(X=x,pad_to=pad_to) for x in X_ts])
    else:
        X_packed = [torch.stack([pad_zeros(X=x[0],pad_to=pad_to) for x in X_ts]), torch.tensor([x[1] for x in X_ts])]
    Y_packed = torch.tensor(np.array([Y for X,Y in ds_zls]))
    
    return X_packed , Y_packed



class mask_reader(Dataset):
    def __init__(self,npy_path):
        """
        read the mask A549 sequence and with real sequence
        """
        self.data_set = np.load(npy_path)
        self.X = self.data_set[:,0]
        self.Y = self.data_set[:,1]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return (self.X[index,:,:],self.Y[index,:,:])

class mix_dataset(Dataset):
    def __init__(self):
        # dataset_ls = [mask_reader(os.path.join(data_dir,'mix_data',"mix_%s.npy"%set)) for set in ['train','val','test']]
        raise ValueError("the dataset is not defined")

class mask_dataset(Dataset):
    def __init__(self):
        raise ValueError("the dataset is not defined")

class ribo_dataset(Dataset):
    
    def __init__(self,DF, pad_to, trunc_len=50, seq_col='utr',
                      aux_columns='TE_count', other_input_columns=None):
        """
        Dataset to trancate sequence and return in one-hot encoding way
        `dataset(DF,pad_to,trunc_len=50,seq_col='utr')`
        ...DF: the dataframe contain sequence and its meta-info
        ...pad_to: final size of the output tensor
        ...trunc_len: maximum sequence to retain. number of nt preceding AUG
        ...seq_col : which col of the DF contain sequence to convert
        """
        DF[seq_col] = DF[seq_col].astype(str)
        self.df = DF
        self.pad_to = pad_to
        self.trunc_len = 0 if trunc_len is None else trunc_len
        
        # X and Y
        self.seqs = self.df.loc[:,seq_col].values
        self.other_input_columns = other_input_columns
        self.Y = self.df.loc[:,aux_columns].values
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,i):
        seq = self.seqs[i]
        x_padded = self.seq_chunk_N_oh(seq)
        input = x_padded
        if self.other_input_columns is not None:
                input = [x_padded]
                for col in self.other_input_columns:
                    input.append(self.df.loc[:,col].values[i]) 
        y = self.Y[i]
        return input,y
    
    def seq_chunk_N_oh(self,seq):
        """
        truncate the sequence and encode in one hot
        """
        if (len(seq) >  self.trunc_len)&(self.trunc_len >0):
            seq = seq[-1* self.trunc_len:]
        
        X = one_hot(seq)
        X = torch.tensor(X)

        # X_padded = pad_zeros(X, self.pad_to)
        
        return X.float()
        
        
class MTL_dataset(Dataset):
    def __init__(self, DF, pad_to=100, seq_col='utr', aux_columns=None,
                other_input_columns=None, trunc_len=None):
        """
        the dataset for Multi-task learning, will return sequence in one-hot encoded version, together with some auxilary task label
        arguments:
        ...csv_path: abs path of csv file, column `utr` should be in the csv
        ...pad_to : maximum length of the sequences
        ...columns : list  contains what  axuslary task label will be 
        """
        self.pad_to = pad_to
        self.trunc_len = trunc_len
        DF[seq_col] = DF[seq_col].astype(str)
        self.df = DF     # read Df
        self.seqs = self.df[seq_col].values       # take out all the sequence in DF
        self.columns = aux_columns
        self.other_input_columns = other_input_columns
        
        assert trunc_len == None, "MTL dataset does not support argument trunc len"
                
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,index):
        seq = self.seqs[index]        # sequence: str of len 25~100
        
        X = one_hot(seq).astype(float)       # seq_oh : np array, one hot encoding sequence 
        # X = pad_zeros(X)         # X  : torch.tensor , zero padded to 100
        
        if self.columns == None:
            # which means no auxilary label is needed
            item = X ,X
        elif (type(self.columns) == list) & (len(self.columns)!=0):
            # return what's in columns
            aux_labels = self.df.loc[:,self.columns].values[index]
            # if len(self.columns) == 1:
            #     aux_labels = aux_labels.reshape(-1,1)
            item = X ,aux_labels

            if self.other_input_columns is not None:
                input = []
                for col in self.other_input_columns:
                    input.append(self.df.loc[:,col].values[index]) 
                item = (X, input),aux_labels

        return item   

class kmer_scan_dataset(Dataset):
    def __init__(self, DF, seq_col, kmer_size, aux_columns=None, *args):
        super().__init__()
        self.df = DF
        self.seqs = DF[seq_col].values
        self.rls = DF[aux_columns].values
        self.k = kmer_size
        self.kmer_sets = self.create_kmer()
        self.kmer_lookup = { kmer : i for i, kmer in enumerate(self.kmer_sets) }

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        seq  = self.seqs[index]
        y = self.rls[index]

        x = self.scan_kmer(seq)
        return torch.tensor(x.astype(float)), torch.tensor(y)
    
    def scan_kmer(self, seq):

        mat = np.zeros((len(self.kmer_lookup), len(seq)-self.k))
        for i in range(0, len(seq)-self.k):
            seq_let = seq[i:i+self.k].upper()
            mat[self.kmer_lookup[seq_let], i]  = 1

        return mat

    
    def create_kmer(self):
        all_kmer = {0:['']}
        k = 1
        while k <= self.k:
            k_mer = []  # 1 ; 4 ; 2: 4**2 ...
            for source in all_kmer[k-1]:
                k_mer += [source + base for base in ['A','C','G','T']]

            assert len(k_mer) == 4**k, f"new kmers {len(k_mer)}, not equal to {4**k}"
            all_kmer[k] = k_mer
            k += 1

        all_kmer.pop(0)
        return all_kmer[self.k]




class FASTA_dataset(MTL_dataset):
    def __init__(self,DF, pad_to,trunc_len, seq_col, aux_columns, other_input_columns):
        super().__init__(DF=DF, pad_to=pad_to, trunc_len=trunc_len, seq_col=seq_col, 
                         aux_columns=aux_columns, other_input_columns=other_input_columns)

def get_splited_dataloader(dataset_func, df_ls, ratio:list, batch_size, shuffle, pad_to, seed=42,return_dataset=False, num_workers=4):
    """
    split the total dataset into train val test, and return in a DataLoader (train_loader,val_loader,test_loader) 
    dataset : the defined <UTR_dataset>
    ratio : the ratio of train : val : test
    batch_size : int
    """

    #  determined set-ls
    
    set_ls = [dataset_func(df) for df in df_ls]
    if return_dataset:
        return set_ls
    
    if pad_to == 0 :
        # automatic padding to `8x -1`
        # and we will pad the sequence of similar length with bucket sampler
        col_fn = lambda x: pack_seq(x,0)
        sampler_ls = [{"batch_sampler":Bucket_Sampler(df,batch_size=batch_size,drop_last=True),
                       "num_workers":num_workers,
                       "drop_last":False,
                       "collate_fn":col_fn} for df in df_ls]
        #  wrap dataset to dataloader
        loader_ls = [DataLoader(subset,**kwargs) for subset,kwargs in zip(set_ls,sampler_ls)]
    else:
        col_fn = lambda x: pack_seq(x,pad_to)
        loaderargs = {"batch_size": batch_size,
                        "generator":torch.Generator().manual_seed(seed),
                        "drop_last":False ,
                        "num_workers":num_workers,
                        "shuffle":shuffle,
                        "collate_fn":col_fn}
        #  wrap dataset to dataloader
        loader_ls = [DataLoader(subset,**loaderargs) for subset in set_ls]
        
    if len(loader_ls) == 2:
        # a complement of empty test set
        loader_ls.append(None) 
        
    return loader_ls

def split_DF(data_path,split_like,ratio, kfold_cv, kfold_index=None,seed=42):
    
    class _cf_data(object):
        def __init__(self,data_path):
            self.path = data_path
            self.is_fasta = data_path.endswith('.fasta')
            self.__read__()
    
        def __read__(self):
            if self.is_fasta:
                self.data = list(SeqIO.parse(self.path,'fasta'))
            else:
                self.data = pd.read_csv(os.path.join(data_dir,self.path),low_memory=False)
        
        def __len__(self):
            return len(self.data)
                
        def _slice_(self, indices):
            if isinstance(self.data, list) & self.is_fasta:
                return np.array(self.data)[indices]
            elif isinstance(self.data, pd.DataFrame):
                return self.data.iloc[indices]
            
        
    if kfold_cv == True:
        full_df = pd.read_csv(os.path.join(data_dir,data_path),low_memory=False)
        # K-fold CV : 8:1:1 for each partition
        df_ls = KFold_df_split(full_df,kfold_index, seed)
        return df_ls
            
    if kfold_cv == 'train_val':
        train_val, test = [_cf_data(data_path).data for data_path in split_like]
        # K-fold CV : 8:1:1 for each partition
        train, val = KFold_1_split(train_val,kfold_index, seed)
        df_ls = [train, val , test]
    
    elif type(split_like) == list:
        df_ls = [_cf_data(data_path).data for data_path in split_like]

    elif type(kfold_cv) == str:
        full_df = pd.read_csv(os.path.join(data_dir,data_path),low_memory=False)
        assert kfold_cv in full_df.columns
        kfold_index = str(kfold_index) 
        test = full_df.query(f"`{kfold_cv}` == @kfold_index")
        train_val = full_df.query(f"`{kfold_cv}` != @kfold_index")
        # chrom "18"
        # train, val = train_test_split(train_val, test_size=0.05)
        val_index= "18"
        val = train_val.query(f"`{kfold_cv}` == @val_index")
        train = train_val.query(f"`{kfold_cv}` != @val_index")
        df_ls = [train, val, test]
    
    else:
        full_df = _cf_data(data_path)
        # POPEN.ratio will determine train :val :test ratio
        total_len = len(full_df)
        lengths = [int(total_len*sub_ratio) for sub_ratio in ratio[:-1]]
        lengths.append(total_len-sum(lengths))         # make sure the sum of length is the total len

        set_ls = random_split(full_df,lengths,generator=torch.Generator().manual_seed(seed)) 
        df_ls = [full_df._slice_(subset.indices) for subset in set_ls] # df.iloc [ idx ]
        
    return df_ls
    
def KFold_df_split(df,K,seed,**kfoldargs):
    """
    split the dataset DF in a ratio of 8:1:1 , train:val:test in the framework of  K-fold CV 
    set random seed = 43
    arguments:
    df : the `pd.DataFrame` object containing all data info
    K : [0,4] , the index of subfold 
    """
    
    # K-fold partition : n_splits=5
    fold_index = list(KFold(10,shuffle=True,random_state=seed).split(df))
    train_val_index, test_index = fold_index[K]  
    # the first 4/5 part of it is train set
     
    # index the df
    train_val_df = df.iloc[train_val_index]
    test_df = df.iloc[test_index]
    
    # the remaining 1/5 data will further break into val and test
    train_df, val_df = train_test_split(train_val_df,test_size=0.05,random_state=seed)
    
    return [train_df,val_df,test_df]   
    
def KFold_1_split(df,K,seed,**kfoldargs):
    """
    for dataset with standard test set, where only train_val can be splited
    """
    fold_index = list(KFold(10,shuffle=True,random_state=seed).split(df))
    train_index, val_index = fold_index[K]  
    # the first 4/5 part of it is train set
     
    # index the df
    train = df.iloc[train_index]
    val = df.iloc[val_index]

    return train , val
    
def get_dataloader(POPEN):
    """
    wrapper
    """
    try:
        seed = POPEN.seed
    except:
        seed = 42
    print("random seed:",seed)
    df_ls = split_DF(POPEN.csv_path,POPEN.split_like,POPEN.train_test_ratio,POPEN.kfold_cv,POPEN.kfold_index,seed=seed)
    
    
    DS_Class = eval(POPEN.dataset+"_dataset")
         
    dataset_func = lambda x : DS_Class(x,pad_to=POPEN.pad_to,trunc_len=POPEN.trunc_len,
                                         seq_col=POPEN.seq_col, aux_columns=POPEN.aux_task_columns,
                                         other_input_columns=POPEN.other_input_columns)
    
    try:
        num_workers = POPEN.num_workers
    except:
        print("num_workers not defined in POPEN, set to 4")
        num_workers = 4
    loader_ls = get_splited_dataloader(dataset_func, df_ls,ratio=POPEN.train_test_ratio,
                                        batch_size=POPEN.batch_size, shuffle=POPEN.shuffle,
                                        pad_to=POPEN.pad_to, seed=seed, num_workers=num_workers) # new function
        
    return loader_ls 