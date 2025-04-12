import os, sys
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


###### data ######
def get_dls(csv_name, kernel_size, seed):
    train_val = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_train_val.csv"))
    MPA_U_test = pd.read_csv(os.path.join(utils.data_dir, f"{csv_name}_test.csv"))

    MPA_U_val = train_val.sample(frac=0.1, random_state=seed)
    MPA_U_train = pd.concat([train_val, MPA_U_val]).drop_duplicates(keep=False)

    # dataset
    train_DS = reader.MTL_dataset(MPA_U_train, seq_col='utr', aux_columns=['rl'])
    val_DS = reader.MTL_dataset(MPA_U_val, seq_col='utr', aux_columns=['rl'])
    test_DS = reader.MTL_dataset(MPA_U_test, seq_col='utr', aux_columns=['rl'])

    # dataloader
    train_dl = DataLoader(train_DS, batch_size = 64, shuffle=True)
    val_dl = DataLoader(test_DS, batch_size = 64, shuffle=False)
    test_dl = DataLoader(test_DS, batch_size = 64, shuffle=False)
    return train_dl, val_dl, test_dl
##################


# PyTorch Light model #
class Onelayer_CNN(pl.LightningModule):
    def __init__(self, kernel_size):
        super().__init__()

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

        self.conv_layer = nn.Sequential(nn.Conv1d(4, 256, kernel_size),
                                    nn.BatchNorm1d(256),
                                    nn.Mish())
        
        tower = { f"GRU_layer" : nn.GRU(input_size=256, hidden_size=128,
                                      num_layers=2,batch_first=True),
                 f"fc_out" : nn.Linear(128, 1), 
                }

        self.tower = nn.ModuleDict(tower)

    def forward(self, x):
        if x.shape[1] != 4:
            x = x.transpose(1,2)
        Z = self.conv_layer(x)
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower['GRU_layer'](Z_t)
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
    
    ####  hyper-params  ####
    csv_name = sys.argv[1]
    kernel_size = sys.argv[2]
    seed = 41
    
    ################
    train_dl, val_dl, test_dl = get_dls(csv_name, kernel_size, seed)
    
    model = Onelayer_CNN(int(kernel_size))


    # training
    trainer = pl.Trainer(gpus=1, num_processes=8,
                         default_root_dir="/ssd/users/wergillius/Project/MTtrans/evaluation/one_layer_logs",
                         limit_train_batches=0.5, max_epochs=60, 
                         callbacks=[callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer.fit(model, train_dl, val_dl)
    trainer.test(test_dl)