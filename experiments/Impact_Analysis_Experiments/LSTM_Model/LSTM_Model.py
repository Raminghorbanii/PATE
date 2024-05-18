#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np




######### LSTM Models


class LSTM(nn.Module):
    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.cfg = cfg

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Dropout layer
        self.dropout1 = nn.Dropout(cfg.model.dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=cfg.model.hidden_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=1,
            batch_first=True
        )


        # Layer to compute the reconstruction
        self.fc_reconstr = nn.Linear(cfg.model.hidden_size, cfg.model.input_size)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        dropped1 = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(dropped1)

        dec_out = self.fc_reconstr(lstm2_out)
        
        return dec_out, lstm1_out, None




