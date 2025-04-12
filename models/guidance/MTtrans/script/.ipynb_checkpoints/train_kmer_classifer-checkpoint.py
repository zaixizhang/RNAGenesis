import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
import pytorch_lightning as pl
from torch.nn import functional as F
import PATH, utils
import torch
from torch import nn, optim
from models import reader
from models.popen import Auto_popen
import pandas as pd
import numpy as np
from collections import OrderedDict
import torchmetrics 
from torch.utils.data import DataLoader
from pytorch_lightning import callbacks 
import argparse


###### data ######
def get_kmer_input_shape(csv_name, kernel_size):
    MPA_U_test = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_test.csv"))

    # dataset
    test_DS = reader.kmer_scan_dataset(MPA_U_test, seq_col='utr', 
                                       kmer_size=kernel_size, aux_columns='rl')
    return np.multiply(*test_DS[0][0].shape)

def get_kmer_dls(csv_name, kernel_size, seed, seq_col, label_col):
    train_val = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_train_val.csv"))
    MPA_U_test = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_test.csv"))

    MPA_U_val = train_val.sample(frac=0.1, random_state=seed)
    MPA_U_train = pd.concat([train_val, MPA_U_val]).drop_duplicates(keep=False)

    # dataset
    train_DS = reader.kmer_scan_dataset(MPA_U_train, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)
    val_DS = reader.kmer_scan_dataset(MPA_U_val, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)
    test_DS = reader.kmer_scan_dataset(MPA_U_test, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)

    # dataloader
    train_dl = DataLoader(train_DS, batch_size = 64, shuffle=True)
    val_dl = DataLoader(test_DS, batch_size = 64, shuffle=False)
    test_dl = DataLoader(test_DS, batch_size = 64, shuffle=False)
    return train_dl, val_dl, test_dl

##################
# PyTorch Light model #
class mlp_models(pl.LightningModule):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        n_layer = len(dims) - 1 
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        # add layers
        nns = []
        i = 1
        for in_dim , out_dim in zip(dims[:-1], dims[1:]):
            nns.append( (f"Linear_{i}", nn.Linear(in_dim, out_dim)) )
            if i < n_layer:
                nns +=  [(f"BN_{i}", nn.BatchNorm1d(out_dim)), (f"act_{i}", nn.Mish()) ]
            i += 1
    
        self.network = nn.Sequential(OrderedDict(nns))

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        b = x.shape[0]
        x = x.view(b, -1).float()
        y = y.view(b,).float()
        y_hat = self.network(x).view(b,)
        loss = F.mse_loss(y_hat, y.float())
        r2 = self.train_r2(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_r2)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        b = x.shape[0]
        x = x.view(b, -1).float()
        y = y.view(b,).float()
        y_hat = self.network(x).view(b,)
        
        loss = F.mse_loss(y_hat, y.float())
        r2 = self.val_r2(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_r2)

# PyTorch Light model #
class rnn_models(pl.LightningModule):
    def __init__(self, k):
        super().__init__()
        
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

       
        tower = { f"GRU_layer" : nn.GRU(input_size=4**k, hidden_size=128,
                                      num_layers=2,batch_first=True),
                 f"fc_out" : nn.Linear(128, 1), 
                }
    
        self.tower = nn.ModuleDict(tower)

    def forward(self, x):
        
        x = x.transpose(1,2) # B C L -> B L C
        h_prim,(c1,c2) = self.tower['GRU_layer'](x)
        out = self.tower['fc_out'](c2)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        b = x.shape[0]
        x = x.float()
        y = y.view(b,).float()
        y_hat = self.forward(x).view(b,)
        
        loss = F.mse_loss(y_hat, y)
        r2 = self.train_r2(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_r2)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        b = x.shape[0]
        x = x.float()
        y = y.view(b,).float()
        y_hat = self.forward(x).view(b,)
        
        loss = F.mse_loss(y_hat, y)
        self.val_r2(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_r2)
        
################
if __name__ == '__main__':

    csv_name = sys.argv[1]
    kmer_size = sys.argv[2]
    hidden = sys.argv[3]
    seq_col = sys.argv[4]
    label_col = sys.argv[5]
    
    
    train_dl, val_dl, test_dl = get_kmer_dls((csv_name, kernel_size, seed, seq_col, label_col))
    

    ####  hyper-params  ####
    Input_length = np.multiply(*train_dl.dataset[0][0].shape)
    dims = [Input_length] + hidden + [1]
    
    model = rnn_models
    ################

    # training
    trainer = pl.Trainer(gpus=1, num_processes=8,
                         default_root_dir="/ssd/users/wergillius/Project/MTtrans/evaluation/Kmer_results",
                         limit_train_batches=0.5, max_epochs=6, 
                         callbacks=[callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer.fit(model, train_dl, val_dl)
    trainer.test(test_dl)