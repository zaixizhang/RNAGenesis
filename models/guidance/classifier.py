import torch.nn as nn
import torch

import os 
from util import XDG_CACHE_HOME
os.environ['XDG_CACHE_HOME'] = XDG_CACHE_HOME # (Optional) to specify where to store the huggingface model/ dataset caches. 

import sys 
sys.path.append("models/autoencoder/encoder") # for the dependencies of UTR-LM
from models.autoencoder.encdec import EncDec
from util import load_model_ckpt, instantiate_from_config
import fm
from ..autoencoder.encoder.esm import Alphabet

class rna_type_classifier(nn.Module):
    def __init__(self, encdec_checkpoint, classifier_config, freeze_encdec=True):
        super(rna_type_classifier, self).__init__()
        self.encdec_model = EncDec.from_pretrained(encdec_checkpoint)
        if freeze_encdec:
            self.encdec_model.requires_grad_(False)
        assert classifier_config is not None
        self.classifier = instantiate_from_config(classifier_config)

    def forward(self, input_ids, attention_mask):
        latent = self.encdec_model.get_latent(input_ids, attention_mask)
        return self.classifier(latent)

class DataCollatorEncCls:
    def __init__(self, class_labels, rna_encoder_type="rna-fm"): #, rna_tokenizer_config="tokenizer_rna"):
        # self.class_labels_idx = {}
        # start_idx = 0
        # if "others" in class_labels:
        #     self.class_labels_idx['others'] = 0
        #     class_labels.remove('others')
        #     start_idx = 1
        # class_labels = sorted(class_labels)
        # for idx, label in enumerate(class_labels):
        #     self.class_labels_idx[label] = idx + start_idx
        # self.class_labels_idx = {label: idx for idx, label in enumerate(class_labels)}
        # self.num_classes = len(self.class_labels_idx)

        self.num_classes = len(class_labels)
        assert rna_encoder_type is not None
        if rna_encoder_type == "rna-fm":
            _, alphabet = fm.pretrained.rna_fm_t12()
        elif rna_encoder_type == "utr-lm":
            # change from "AGCT" to "AGCU"
            alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
            assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
        self.enc_pad_token = alphabet.padding_idx
  

    def __call__(self, instances):
        # merge and pad seq.
        input_ids = [torch.tensor(ins['input_ids'], dtype=torch.long) for ins in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value = self.enc_pad_token)
        attention_mask = input_ids.ne(self.enc_pad_token)
        # class_labels = torch.tensor([self.class_labels_idx[ins['class_label_x']] for ins in instances], dtype=torch.long)
        class_labels = torch.tensor([ins['class_label_x'] for ins in instances], dtype=torch.long)
        return dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                class_labels = class_labels
                )

##### latent classifier #####
class fc_classifier(nn.Module):
    def __init__(self, num_classes, preprocess="flatten", emb_dim= 160, token_num = 32, model_path=None):
        super(fc_classifier, self).__init__()
        self.preprocess = preprocess
        self.num_classes = num_classes
        if preprocess == "flatten":
            input_dim = emb_dim * token_num
        elif preprocess == "mean":
            input_dim = emb_dim
        self.fc = nn.Linear(input_dim, num_classes)
        self.ckpt_loaded = False
        if model_path is not None:
            load_model_ckpt(self, model_path, map_location="cpu")
            self.ckpt_loaded = True

    def forward(self, x):
        if self.preprocess == "flatten":
            x = x.view(x.size(0), -1)
        elif self.preprocess == "mean":
            x = x.mean(dim=1)
        return self.fc(x)
    

class transformer_classifier(nn.Module):
    def __init__(
            self, 
            num_classes,
            model_path=None,
            ### args for transformer
            embedding_dim=80, 
            nhead = 16, # 8
            num_layers = 12, #6
            dropout = 0, #0.1, 
            ):
        super(transformer_classifier, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Transformer Encoder Layer and Transformer Encoder
        self.layer1 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layer1, num_layers=num_layers)
        self.conv1=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
        # Output Layer
        self.output = nn.Linear(embedding_dim, num_classes)
        self.ckpt_loaded = False
        if model_path is not None:
            load_model_ckpt(self, model_path, map_location="cpu")
            self.ckpt_loaded = True
        self.device = "cpu"

    def forward(self, x):     
        x = self.transformer(x)  # Transformer outputs [batch, query, embedding_dim]
        x = x.permute(0, 2, 1)    # Change the shape to [batch, embedding, num_channels]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=1)
        x = x.view(-1, self.embedding_dim)  # Flatten the output for the linear layer
        return self.output(x)
    

class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        dilation=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        groups = 1,
    ):
        super(ResBlock, self).__init__()    
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups)       
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, padding=dilation, bias=False, groups=groups)

        if stride > 1 or out_planes != in_planes: 
            self.downsample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None
            
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    

class ResNet_classifier(nn.Module):
    def __init__(self, 
        num_classes,
        model_path=None,
        embedding_dim=160, 
        seq_len=16,
        dropout=0.0,
        stride_down = 2,
        groups = 1,
        add_dropout = False,
        second_last_stride = 2,
        ):
        super(ResNet_classifier, self).__init__()
        self.num_classes = num_classes

        self.embedding_dim=embedding_dim
        main_planes = embedding_dim * 2
        self.main_planes = main_planes 
        self.dropout = dropout
        in_planes = embedding_dim
        out_planes = num_classes

        blocks = [
            nn.ConvTranspose1d(in_planes, main_planes, kernel_size=3,stride=2),
        ]
        strides = [stride_down, 1, stride_down, 1, second_last_stride, 1]
        for i in range(6):
            blocks.append(
                ResBlock(main_planes * 1, main_planes * 1, stride=strides[i], dilation=1, conv_layer=nn.Conv1d, groups = groups,
                        norm_layer=nn.BatchNorm1d)
            ) 
            if add_dropout:
                blocks.append(nn.Dropout(dropout))         

        self.embcnn=nn.Sequential(*blocks)
        if stride_down == 1:
            self.fc1 = nn.Linear(2*seq_len+1, 1)
        elif stride_down == 2:
            if second_last_stride == 1:
                self.fc1 = nn.Linear(int(seq_len/2)+1, 1)
            elif second_last_stride == 2:
                self.fc1 = nn.Linear(int(seq_len/4)+1, 1)
            # self.fc1 = nn.Linear(int(seq_len/2)+1, 1)

        self.relu=nn.ReLU()
        self.predictor = nn.Linear(main_planes * 1, out_planes)
        
        self.ckpt_loaded = False
        if model_path is not None:
            load_model_ckpt(self, model_path, map_location="cpu")
            self.ckpt_loaded = True 

    def forward(self, x):
    # def forward(self, x,mask=None):
    #     if mask is not None:
    #         mask = mask.unsqueeze(-1).expand_as(x)
    #         x = x.masked_fill(mask, 0)  
        x = x.permute(0, 2, 1)
        x = self.embcnn(x)
        batch_size,  feature_dim, seq_len = x.shape
        x= x.view(-1, seq_len)
        x=self.fc1(x)
        x=self.relu(x)
        x = x.view(batch_size, feature_dim)
        x = self.predictor(x)
        return x

class CNN_classifier(nn.Module):
    def __init__(self, num_classes, in_planes, mid_planes, model_path=None):
        super(CNN_classifier, self).__init__()
        self.num_classes = num_classes
        norm_layer = nn.LayerNorm
        # conv_layer = nn.Conv1d
        self.bn1 = norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv_layer(in_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups)       
        self.bn2 = norm_layer(mid_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = conv_layer(mid_planes, mid_planes, kernel_size=3, padding=dilation, bias=False, groups=groups)
        self.conv1=nn.Conv1d(in_channels=in_planes,out_channels=mid_planes,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv1d(in_channels=mid_planes,out_channels=mid_planes,kernel_size=3,stride=1,padding=1)
        self.output = nn.Linear(mid_planes, num_classes)
        self.ckpt_loaded = False
        if model_path is not None:
            load_model_ckpt(self, model_path, map_location="cpu")
            self.ckpt_loaded = True 

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = out.permute(0, 2, 1) # [batch, embedding, num_channels]

        out = self.conv1(out)  

        out = out.permute(0, 2, 1) # [batch, num_channels, embedding]
        out = self.bn2(out)
        out = self.relu(out)
        out = out.permute(0, 2, 1) # [batch, embedding, num_channels]

        out = self.conv2(out)
        out = out.mean(dim=2)
        out = self.output(out)
        return out

class MLP_classifier(nn.Module):
    def __init__(self, num_classes, preprocess="flatten", emb_dim=160, token_num=32,
                 hidden_dims=[512, 256], dropout=0.5, use_batchnorm=False, model_path=None):
        super(MLP_classifier, self).__init__()
        self.preprocess = preprocess
        self.num_classes = num_classes

        # for guidance
        self.classifier_type = "latent"
        self.guidance_type = "classifier"

        if preprocess == "flatten":
            input_dim = emb_dim * token_num
        elif preprocess == "mean":
            input_dim = emb_dim
        else:
            raise ValueError("Unsupported preprocess type: {}".format(preprocess))

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        self.ckpt_loaded = False
        if model_path is not None:
            load_model_ckpt(self, model_path, map_location="cpu")
            self.ckpt_loaded = True

    def forward(self, x):
        if self.preprocess == "flatten":
            x = x.view(x.size(0), -1)
        elif self.preprocess == "mean":
            x = x.mean(dim=1)
        return self.mlp(x)
