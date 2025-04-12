import torch
import torch.nn as nn
import numpy as np
import sys
from util import load_model_ckpt, filter_seqs, instantiate_from_config
import os
from models.autoencoder.encdec import EncDec
import fm
from models.autoencoder.encoder.esm import Alphabet
from models.guidance.framepool import load_framepool
sys.path.append("models/guidance/MTtrans")

## MTtrans [TE Reward Model]
class MTtrans:
    reward_type = "TE"
    guidance_type = "seq"
    def __init__(self, model_path="models/guidance/MTtrans/checkpoint/sH_293-model_best_cv1.pth", max_len=128):
        self.model = torch.load(model_path,map_location="cpu")['state_dict']  
        # self.min_len = None
        self.max_len = max_len #TODO: check if this is the correct length
        # self.nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
        #                 'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
        #                 'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}
        self.device = "cpu"

    # def forward(self, x):
    def __call__(self, x):
        return self.model(x)
    
    def one_hot_motif(self, seq, length=-1, complementary=False):
        """
        one_hot encoding on sequence
        params:
            seq: sequence of tokens [string]
            length: length of the sequence [int], if -1, use the length of the sequence, default -1 (or 128)
            complementary: encode nucleatide into complementary one # [T, G, C, A]
        """
        
        if length == -1:
            length = len(seq)
        # setting
        seq = list(seq.replace("U","T"))
        # seq_len = len(seq)
        complementary = -1 if complementary else 1
        # compose dict
        keys = ['A', 'C', 'G', 'T'][::complementary]
        oh_dict = {keys[i]:i for i in range(4)}
        # array
        oh_array = np.zeros((length,4),dtype=np.double)
        for i,C in enumerate(seq):
            try:
                oh_array[i,oh_dict[C]]=1 # TODO: add look-up table for transistion of non-ACGT tokens
            except:
                continue      # for nucleotide that are not in A C G T   
        return oh_array

    def one_hot_all_motif(self, seqs):
        '''
        seqs: sequence of tokens [list of sequences]
        '''
        max_len = np.max([len(seq) for seq in seqs]) 
        x = [self.one_hot_motif(seqs[i], length = max_len) for i in range(len(seqs))]
        x = torch.tensor(np.array(x,dtype=np.float32))    

        # x = np.array([self.encode_seq_framepool(seq) for seq in seqs])
        # x = torch.from_numpy(x).float()
        return x


    def prepare_input_for_reward_model(self, seqs, do_filter_seqs=False, **kwargs):
        '''
        Prepare input for the reward model
        seqs: sequence of tokens [list of sequences]
        '''
        if do_filter_seqs:
            seqs = filter_seqs(seqs, **kwargs)
        x = self.one_hot_all_motif(seqs)
        return x


    # def encode_seq_framepool(self, seq):
    #     '''
    #     Convert a single sequence into one-hot encoding
    #     seq: sequence of tokens [string]
    #     '''
    #     # print(seq)
    #     length = len(seq)
    #     if self.max_len > 0 and self.min_len is None:
    #         padding_needed = self.max_len - length
    #         seq = "N"*padding_needed + seq
    #     if self.min_len is not None:
    #         if len(seq) < self.min_len:
    #             seq = "N"*(self.min_len - len(seq)) + seq
    #         if len(seq) > self.min_len:
    #             seq = seq[(len(seq) - self.min_len):]
    #     seq = seq.lower()
    #     one_hot = np.array([self.nuc_dict[x] for x in seq]) # get stacked on top of each other
    #     return one_hot

class Transformer_Reward_model_qt(nn.Module):
    reward_type = "TE"
    guidance_type = "latent"
    def __init__(
            self, model_path,
            embedding_dim=80, 
            nhead = 16, # 8
            num_layers = 12, #6
            dropout = 0.1, 
            reward_type=None,
            ):
        super(Transformer_Reward_model_qt, self).__init__()
        if reward_type is not None:
            self.reward_type = reward_type
        self.embedding_dim = embedding_dim
        # self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Layer and Transformer Encoder
        self.layer1 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layer1, num_layers=num_layers)
        self.conv1=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
        # Output Layer
        self.output = nn.Linear(embedding_dim, 1)
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        load_model_ckpt(self, model_path, map_location="cpu")
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

    # def from_pretrained(self, path):         
    #     self.load_state_dict(torch.load(path))
    #     return self   

# class Transformer_Reward_model_qt_v2(Transformer_Reward_model_qt):
#     def __init__(self, model_path, embedding_dim=80, dropout=0.1):
#         super(Transformer_Reward_model_qt_v2, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.dropout = nn.Dropout(dropout)

#         # Transformer Encoder Layer and Transformer Encoder
#         self.layer1 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dropout=dropout, batch_first=True)
#         self.transformer = nn.TransformerEncoder(self.layer1, num_layers=6)
#         self.conv1=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
#         self.conv2=nn.Conv1d(in_channels=embedding_dim,out_channels=embedding_dim,kernel_size=3,stride=1,padding=1)
#         # Output Layer
#         self.output = nn.Linear(embedding_dim, 1)
#         assert os.path.exists(model_path), f"Model path {model_path} does not exist"
#         load_model_ckpt(self, model_path, map_location="cpu")
#         self.device = "cpu"


class Seq_Reward_model_latent_qt(nn.Module):
    # reward_type = "TE"
    guidance_type = "seq"
    def __init__(
            self, 
            encdec_checkpoint,
            reward_model_path=None,
            reward_type="TE",
            reward_model_config=None,
            **kwargs
            ):
        super(Seq_Reward_model_latent_qt, self).__init__()
        self.encdec = EncDec.from_pretrained(encdec_checkpoint)
        self.encoder_type = self.encdec.rna_encoder_type
        self.build_enc_tokenizer()

        self.reward_type = reward_type
        if reward_model_config is None:
            assert reward_model_path is not None, "reward_model_path is None"
            if reward_type == "TE":
                self.latent_reward_model = Transformer_Reward_model_qt(
                    reward_model_path, **kwargs)
            elif reward_type == "MRL":
                self.latent_reward_model = RNAFM_ResNet_qt(
                    reward_model_path, **kwargs)
            else:
                raise ValueError(f"Reward type {reward_type} not supported")
        else:
            self.latent_reward_model = instantiate_from_config(reward_model_config)
        assert self.reward_type == self.latent_reward_model.reward_type
        self.device = "cpu"

    def forward(self, x): 
        # x: processed latent vectors    
        return self.latent_reward_model(x)

    def build_enc_tokenizer(self):
        if self.encoder_type == "rna-fm":
            _, alphabet = fm.pretrained.rna_fm_t12()
            # batch_converter = alphabet.get_batch_converter()
        elif self.encoder_type == "utr-lm":
            # change from "AGCT" to "AGCU"
            alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
            assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
        self.batch_converter = alphabet.get_batch_converter()
        self.alphabet = alphabet
        self.enc_pad_token = alphabet.padding_idx

    def preprocess(self, seqs):
        input_ids = []
        for seq in seqs:
            _, _, input_ids_ = self.batch_converter([("rna1", seq)])
            input_id = input_ids_[0]
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value = self.enc_pad_token)
        attention_mask = input_ids.ne(self.enc_pad_token)
        return input_ids, attention_mask

    def prepare_input_for_reward_model(
            self, seqs=None, do_filter_seqs=True, 
            input_ids = None, attention_mask = None,
            **kwargs):
        '''
        Prepare input for the reward model
        seqs: sequence of tokens [list of sequences]
        or input_ids and attention_mask
        '''
        if input_ids is None or attention_mask is None:
            assert seqs is not None, "seqs is None"
            if do_filter_seqs:
                seqs = filter_seqs(seqs, **kwargs)
            input_ids, attention_mask = self.preprocess(seqs)
        latent = self.encdec.get_latent(input_ids.to(self.device), attention_mask.to(self.device))
        return latent
    
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


class RNAFM_ResNet_qt(nn.Module):
    reward_type = "MRL"
    guidance_type = "latent"
    def __init__(self,
        model_path, 
        embedding_dim=160,
        reward_type=None,
        ):
        super(RNAFM_ResNet_qt, self).__init__()
        if self.reward_type is not None:
            self.reward_type = reward_type
        main_planes = 64
        # main_planes = 320
        self.dropout = 0.2
        in_planes = 32
        # in_planes = 160
        out_planes = 1
        self.reduction_module = nn.Linear(embedding_dim, 32)
        self.embcnn=nn.Sequential(
            nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1), 
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),       
        )
        self.predictor = nn.Linear(main_planes * 1, out_planes)
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        load_model_ckpt(self, model_path, map_location="cpu")
        self.device = "cpu"

    def forward(self, x,mask=None):

        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask, 0)
        
        x = self.reduction_module(x)
        x = x.permute(0, 2, 1)
        x = self.embcnn(x)
        if mask is not None:
            mask=mask.permute(0, 2, 1)
            mask = ~mask  # Invert the mask for averaging: 1s for valid, 0s for masked
            mask=mask.float().mean(dim=1, keepdim=True) #use broadcast later too match mask's shape with x's
            #print(mask.shape)
            sum_x = torch.sum(x, dim=2)
            count_x = torch.sum(mask, dim=2)
            avg_x = sum_x / count_x.clamp(min=1)  # Avoid division by zero
            x = avg_x.unsqueeze(-1)  # Reshape if necessary
        else:
            x = x.mean(dim=2)
        x = x.view(-1, 64)
        #x = x.view(-1, 320)
        x = self.predictor(x)
        return x

    # def save_pretrained(self, path):
    #     torch.save(self.state_dict(), path)
    # def from_pretrained(self, path):         
    #     self.load_state_dict(torch.load(path))
    #     return self       

class RNAFM_ResNet_qt_v2(nn.Module):
    reward_type = "MRL"
    guidance_type = "latent"
    def __init__(self, 
            model_path, 
            embedding_dim=160, dropout=0.2,
            main_planes=320, 
            reward_type=None,
            ):
        super(RNAFM_ResNet_qt_v2, self).__init__()
        if reward_type is not None:
            self.reward_type = reward_type
        self.embedding_dim = embedding_dim
        if main_planes is None:
            main_planes = embedding_dim * 2
        self.main_planes = main_planes  
        # #main_planes = 64
        # main_planes = 320
        self.dropout = dropout
        #in_planes = 32
        in_planes = embedding_dim #160
        out_planes = 1
        #self.reduction_module = nn.Linear(embedding_dim, 32)
        self.embcnn=nn.Sequential(
            nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1), 
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            nn.Dropout(dropout),         
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            nn.Dropout(dropout),         
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            nn.Dropout(dropout),           
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),
            nn.Dropout(dropout),    
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),
            nn.Dropout(dropout),   
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            nn.Dropout(dropout),       
        )
        self.predictor = nn.Linear(main_planes * 1, out_planes)
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        load_model_ckpt(self, model_path, map_location="cpu")
        self.device = "cpu"


    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask, 0)
        
        #x = self.reduction_module(x)
        x = x.permute(0, 2, 1)
        x = self.embcnn(x)
        if mask is not None:
            mask=mask.permute(0, 2, 1)
            mask = ~mask  # Invert the mask for averaging: 1s for valid, 0s for masked
            mask=mask.float().mean(dim=1, keepdim=True) #use broadcast later too match mask's shape with x's
            #print(mask.shape)
            sum_x = torch.sum(x, dim=2)
            count_x = torch.sum(mask, dim=2)
            avg_x = sum_x / count_x.clamp(min=1)  # Avoid division by zero
            x = avg_x.unsqueeze(-1)  # Reshape if necessary
        else:
            x = x.mean(dim=2)
        #x = x.view(-1, 64)
        x = x.view(-1, self.main_planes)
        x = self.predictor(x)
        return x

    # def save_pretrained(self, path):
    #     torch.save(self.state_dict(), path)
    # def from_pretrained(self, path):         
    #     self.load_state_dict(torch.load(path))
    #     return self 

# back-up
# class RNAFM_ResNet_qt_v3(nn.Module):
#     reward_type = "MRL"
#     guidance_type = "latent"
#     def __init__(self, 
#         model_path,
#         embedding_dim=160, 
#         seq_len=16,
#         main_planes=None, #
#         dropout=0.0, # 0.2
#         reward_type=None,
#         ):
#         super(RNAFM_ResNet_qt_v3, self).__init__()
#         if reward_type is not None:
#             self.reward_type = reward_type
#         self.embedding_dim=embedding_dim
#         if main_planes is None:
#             main_planes = embedding_dim * 2
#         self.main_planes = main_planes 
#         self.dropout = dropout
#         in_planes = embedding_dim
#         out_planes = 1

#         self.embcnn=nn.Sequential(
#             nn.ConvTranspose1d(in_planes, main_planes, kernel_size=3,stride=2),
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d), 
#             nn.Dropout(dropout),         
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d),  
#             nn.Dropout(dropout),         
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d), 
#             nn.Dropout(dropout),           
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d),
#             nn.Dropout(dropout),    
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d),
#             nn.Dropout(dropout),   
#             ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d, groups = 4,
#                      norm_layer=nn.BatchNorm1d), 
#             nn.Dropout(dropout),       
#         )
#         self.fc1 = nn.Linear(2*seq_len+1, 1)
#         self.relu=nn.ReLU()
#         self.predictor = nn.Linear(main_planes * 1, out_planes)
        
#         assert os.path.exists(model_path), f"Model path {model_path} does not exist"
#         load_model_ckpt(self, model_path, map_location="cpu")
#         self.device = "cpu"

#     def forward(self, x,mask=None):

#         if mask is not None:
#             mask = mask.unsqueeze(-1).expand_as(x)
#             x = x.masked_fill(mask, 0)  
#         #print(x.shape)         
#         x = x.permute(0, 2, 1)
#         x = self.embcnn(x)
#         #print(x.shape)  
#         batch_size,  feature_dim, seq_len = x.shape
#         x= x.view(-1, seq_len)
#         #print(x.shape)  
#         x=self.fc1(x)
#         x=self.relu(x)
#         x = x.view(batch_size, feature_dim)
#         #print(x.shape)  
#         x = self.predictor(x)
#         return x

#     # def save_pretrained(self, path):
#     #     torch.save(self.state_dict(), path)
#     # def from_pretrained(self, path):         
#     #     self.load_state_dict(torch.load(path))
#     #     return self       
    
class RNAFM_ResNet_qt_v3(nn.Module):
    reward_type = "MRL"
    guidance_type = "latent"
    def __init__(self, 
        model_path,
        embedding_dim=160, 
        seq_len=16,
        main_planes=None, #
        dropout=0.0, # 0.2
        reward_type=None,
        #### Update 0727 ######
        stride_down = 1,
        groups = 4,
        add_dropout = True,
        #### Update 0802 ######
        second_last_stride = 1,
        ):
        super(RNAFM_ResNet_qt_v3, self).__init__()
        if reward_type is not None:
            self.reward_type = reward_type
        self.embedding_dim=embedding_dim
        if main_planes is None:
            main_planes = embedding_dim * 2
        self.main_planes = main_planes 
        self.dropout = dropout
        in_planes = embedding_dim
        out_planes = 1

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
        
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        load_model_ckpt(self, model_path, map_location="cpu")
        self.device = "cpu"

    def forward(self, x,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask, 0)  
        x = x.permute(0, 2, 1)
        x = self.embcnn(x)
        batch_size,  feature_dim, seq_len = x.shape
        x= x.view(-1, seq_len)
        x=self.fc1(x)
        x=self.relu(x)
        x = x.view(batch_size, feature_dim)
        x = self.predictor(x)
        return x
    
class framepool:
    reward_type = "MRL"
    guidance_type = "seq"
    def __init__(self, model_path="models/guidance/framepool/utr_model_combined_residual_new.h5"):
        super(framepool).__init__()
        self.model = load_framepool(model_path)
        self.nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
                'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
                'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}
        self.device = "cuda" # the model will be loaded in the GPU

    def __call__(self, x):
        return self.model(x)
    
    def prepare_input_for_reward_model(
            self, seqs, do_filter_seqs=False, **kwargs):
        '''
        Prepare input for the reward model
        seqs: sequence of tokens [list of sequences]
        '''
        if do_filter_seqs:
            seqs = filter_seqs(seqs, **kwargs)
        # x = np.array([self.encode_seq_framepool(seq) for seq in seqs])
        x = self.encode_all_seqs(seqs)
        return x
    
    def encode_all_seqs(self, seqs):
        '''
        seqs: sequence of tokens [list of sequences]
        '''
        max_len = np.max([len(seq) for seq in seqs]) 
        x = [self.encode_seq_framepool(seqs[i], max_len = max_len) for i in range(len(seqs))]
        x = np.array(x)
        # x = tf.convert_to_tensor(x, dtype=tf.float32)  
        x = torch.tensor(x, dtype=torch.float32)
        return x 

    #### Borrowed from UTRGAN #####
    def encode_seq_framepool(self, seq, max_len=128, min_len=None):
        length = len(seq)
        if max_len > 0 and min_len is None:
            padding_needed = max_len - length
            seq = "N"*padding_needed + seq # pad the sequence if padding_needed > 0 else add ""
        if min_len is not None:
            if len(seq) < min_len:
                seq = "N"*(min_len - len(seq)) + seq
            if len(seq) > min_len:
                seq = seq[(len(seq) - min_len):]
        seq = seq.lower()
        one_hot = np.array([self.nuc_dict[x if x in self.nuc_dict else "n"] for x in seq]) # get stacked on top of each other
        return one_hot