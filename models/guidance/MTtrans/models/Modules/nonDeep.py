import os
import sys
sys.path.append(os.path.abspath("../"))
import numpy as np
import pandas as pd
import torch
from torch import nn
import reader


class Kmer_LinReg(nn.Module):
    """
    simple kmer model detect motifs
    """
    def __init__(self, kmer_size, pad_to):
        super().__init__()
        self.k = kmer_size
        self.input_length = pad_to

        # define Conv then replace the parameters
        channel_size = 4**kmer_size 
        self.kmer_conv = nn.Conv1d(4, 1, kmer_size)
        self.custom_conv()

        self.kmer_binarizer = nn.ReLU()

        out_length = self.compute_kmer_outshape()
        self.fc_out = nn.Linear(channel_size * out_length, 1)

    def compute_kmer_outshape(self):
        """
        default stride = 1, pad = 0
        """
        dilation = 1
        padding = 0
        stride = 1
        L_in = self.input_length 
        L_out = 1 + L_in + 2 * padding - dilation * (self.k - 1) - 1 
        return L_out

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
        return all_kmer
    
    def custom_conv(self):

        # only detect 5-mer is kmersize is 5
        # Zhang et al includes shorter kmers in their features
        kmers = self.create_kmer()[self.k]
        
        matrix = [reader.one_hot(kmer).T for kmer in kmers]
        kernels = np.stack(matrix)

        kernels = matrix[0].reshape(1, 4, 3)

        self.kmer_conv.weight = nn.Parameter(torch.from_numpy(kernels).long(), requires_grad=False)


