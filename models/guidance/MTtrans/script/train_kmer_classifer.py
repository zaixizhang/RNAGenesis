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
from sklearn.model_selection import train_test_split
import argparse

################
if __name__ == '__main__':
    global_seed = int(sys.argv[6])
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)


###### data ######
def get_kmer_input_shape(csv_name, kernel_size):
    MPA_U_test = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_test.csv"))

    # dataset
    test_DS = reader.kmer_scan_dataset(MPA_U_test, seq_col='utr', 
                                       kmer_size=kernel_size, aux_columns='rl')
    return np.multiply(*test_DS[0][0].shape)

def get_kmer_dls(csv_name, kernel_size, seed, seq_col, label_col):
    
    print("seed = %s"%seed)
    # train_df, val_df, test_df = reader.split_DF(csv_name, None, [0.8,0.1,0.1], kfold_cv=True, kfold_index=seed,seed=43)
    Seq_df = pd.read_csv(csv_name)
    if Seq_df[label_col].nunique != 2:
        subset_index = Seq_df[label_col].isin([0,1])
        Seq_df = Seq_df[subset_index]


    train_df, val_df, test_df = reader.KFold_df_split(Seq_df.reset_index(), seed)
    # dataset
    train_DS = reader.kmer_scan_dataset(train_df, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)
    val_DS = reader.kmer_scan_dataset(val_df, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)
    test_DS = reader.kmer_scan_dataset(test_df, seq_col=seq_col, kmer_size=kernel_size, aux_columns=label_col)

    # dataloader
    train_dl = DataLoader(train_DS, batch_size = 64, shuffle=True)
    val_dl = DataLoader(val_DS, batch_size = 64, shuffle=False)
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
    def __init__(self, k, hidden=128):
        super().__init__()
        self.save_hyperparameters()
        self.train_F1 = torchmetrics.F1Score()
        self.train_AUROC = torchmetrics.AUROC()
        self.train_ACC = torchmetrics.Accuracy()

        self.val_F1 = torchmetrics.F1Score()
        self.val_AUROC = torchmetrics.AUROC()
        self.val_ACC = torchmetrics.Accuracy()

        self.test_F1 = torchmetrics.F1Score()
        self.test_AUROC = torchmetrics.AUROC()
        self.test_ACC = torchmetrics.Accuracy()
       
        tower = { f"GRU_layer" : nn.GRU(input_size=4**k, hidden_size=hidden,
                                      num_layers=2,batch_first=True),
                 f"fc_out" : nn.Linear(hidden, 1), 
                }
        self.sigmod = nn.Sigmoid()
        self.tower = nn.ModuleDict(tower)

    def forward(self, x):
        
        x = x.transpose(1,2) # B C L -> B L C
        h_prim,(c1,c2) = self.tower['GRU_layer'](x)
        out = self.tower['fc_out'](c2)
        return self.sigmod(out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        b = x.shape[0]
        x = x.float()
        y = y.view(b,)
        y_hat = self.forward(x).view(b,)
        
        loss = F.binary_cross_entropy(y_hat, y.float())
        F1 = self.train_F1(y_hat, y.long())
        auroc = self.train_AUROC(y_hat, y.long())
        acc = self.train_ACC(y_hat, y.long())
        self.log('train_loss', loss)
        self.log('train_f1', self.train_F1)
        self.log('train_AUROC', self.train_AUROC)
        self.log('train_ACC', self.train_ACC)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        b = x.shape[0]
        x = x.float()
        y = y.view(b,)
        y_hat = self.forward(x).view(b,)
        
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.val_F1(y_hat, y.long())
        self.val_AUROC(y_hat, y.long())
        self.val_ACC(y_hat, y.long())
        self.log('val_loss', loss)
        self.log('val_F1', self.val_F1)
        self.log('val_AUROC', self.val_AUROC)
        self.log('val_ACC', self.val_ACC)
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        b = x.shape[0]
        x = x.float()
        y = y.view(b,)
        y_hat = self.forward(x).view(b,)
        
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.test_F1(y_hat, y.long())
        self.test_AUROC(y_hat, y.long())
        self.test_ACC(y_hat, y.long())
        self.log('test_loss', loss)
        self.log('test_F1', self.test_F1)
        self.log('test_AUROC', self.test_AUROC)
        self.log('test_ACC', self.test_ACC)
        
################
if __name__ == '__main__':

    csv_name = sys.argv[1]
    kmer_size = int(sys.argv[2])
    hidden = int(sys.argv[3])
    seq_col = sys.argv[4]
    label_col = sys.argv[5]

    train_dl, val_dl, test_dl = get_kmer_dls(csv_name, kmer_size, global_seed, seq_col, label_col)
    

    ####  hyper-params  ####
    Input_length = np.multiply(*train_dl.dataset[0][0].shape)
    # dims = [Input_length] + hidden + [1]
    
    model = rnn_models(kmer_size, hidden)
    default_root_dir="/data/users/wergillius/UTR_VAE/pth/Kmer_Alan"
    # default_root_dir="/ssd/users/wergillius/Project/MTtrans/evaluation/Kmer_results"
    data_name = os.path.basename(csv_name).split("_")[0]
    log_dir = os.path.join(default_root_dir, f"{data_name}_K{kmer_size}H{hidden}_sd{global_seed}")
    ################

    # training
    trainer = pl.Trainer(accelerator='gpu',devices=1, auto_select_gpus=False, # 
                         default_root_dir=log_dir,  
                         limit_train_batches=0.5, max_epochs=600, 
                         #plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                         callbacks=[
                            callbacks.ModelCheckpoint(monitor="val_loss",save_top_k=1),
                            callbacks.EarlyStopping(monitor="val_F1", mode="max", patience=15)                                    
                         ])

   
    trainer.fit(model, train_dl, val_dl)

    ckpt_dir = os.path.join(trainer.log_dir,'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, [file for file in os.listdir(ckpt_dir) if file.endswith('.ckpt')][0])
    print('model saved to : %s\n'%ckpt_path)
    # trainer.validate(model, test_dl)

    saved_model = rnn_models.load_from_checkpoint(ckpt_path)
    trainer.validate(saved_model, val_dl)
    trainer.test(saved_model, test_dl)

# debugging args:
    # "/data/users/wergillius/UTR_VAE/Alan_dataset/AlanAll_binary_10pctg.csv",
    #                  "3",
    #                  "64",
    #                  "seq",
    #                  "Binary_10pc",
    #                  "1"


    # "/data/users/wergillius/UTR_VAE/Alan_dataset/N600_nAUG_2class.csv",
    #             "3",
    #             "64",
    #             "seq",
    #             "clf_label",
    #             "2"