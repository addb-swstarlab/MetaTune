# Network of Predictive Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import dataset as D
import copy, random
import numpy as np
from scipy.spatial import distance
import logging

class EncoderConv(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, kernel_size, dropout_p, input_dim=1, max_length=22):
        super(EncoderConv, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).cuda()       
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        
        self.knob_embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        
        self.emb2hid = nn.Linear(embed_dim, hidden_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hidden_dim,
                                              out_channels = 2 * hidden_dim,
                                              kernel_size = kernel_size,
                                              padding = (kernel_size -1) // 2)
                                     for _ in range(self.n_layers)])
        
        self.hid2emb = nn.Linear(hidden_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(axis=-1)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        
        src_embedded = self.knob_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        
        embedded = self.dropout(src_embedded + pos_embedded)
        
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1)
        
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        
        conved = self.hid2emb(conved.permute(0, 2, 1))
            
        combined = (conved + embedded) * self.scale
        
        return conved, combined
    
    def _parameterized(self, src, weights):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        
        src_embedded = F.linear(src, weights[0], weights[1])
        pos_embedded = F.embedding(pos, weights[2])
        
        embedded = self.dropout(src_embedded + pos_embedded)
        
        conv_input = F.linear(embedded, weights[3], weights[4])
        conv_input = conv_input.permute(0, 2, 1)
        
        conv_idx = 5
        for _ in range(self.n_layers):
            conved = self.dropout(conv_input)
            conved = F.conv1d(conved, weights[conv_idx], weights[conv_idx+1], padding=(self.kernel_size -1) // 2)
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_idx += 2
            conv_input = conved
            
        conved = F.linear(conved.permute(0, 2, 1), weights[-2], weights[-1])
        
        combined = (conved + embedded) * self.scale
        
        return conved, combined

def DecoderConv():
    pass

def ConvNet():
    pass