#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: idcnn.py
@time:2022/05/06
@description:
"""
from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F


class IDCNN(nn.Module):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=5e-03)
        parser.add_argument("--block", type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--data_path', type=str, default="data/corpus")
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=9e-3)
        parser.add_argument("--char_embedding_size", type=int, default=60)
        parser.add_argument("--experiment", type=bool, default=False)
        return parser

    def __init__(self,
                 token_vocab_size,
                 num_labels,
                 token_embedding_dim=128,
                 cnn_kernel_size=3,
                 cnn_num_filters=128,
                 input_dropout=0.5,
                 middle_dropout=0.2,
                 blocks=1,
                 dilation_l=None,
                 embedding_pad_idx=0,
                 drop_penalty=1e-4
                 ):
        super().__init__()
        if dilation_l is None:
            dilation_l = [1, 2, 1]
        self.num_blocks = blocks
        self.dilation_l = dilation_l
        self.drop_penalty = drop_penalty
        self.num_labels = num_labels
        self.padding_idx = embedding_pad_idx
        self.token_embedding_dim = token_embedding_dim
        self.token_embedding = nn.Embedding(token_vocab_size,
                                            self.token_embedding_dim,
                                            padding_idx=embedding_pad_idx)
        self.filters = cnn_num_filters
        padding_word = int(cnn_kernel_size / 2)
        self.conv0 = nn.Conv1d(in_channels=token_embedding_dim,
                               out_channels=self.filters,
                               kernel_size=cnn_kernel_size,
                               padding=padding_word)
        self.cov_layers = nn.ModuleList([
        nn.Conv1d(in_channels=cnn_num_filters,
                  out_channels=cnn_num_filters,
                  kernel_size=cnn_kernel_size,
                  padding=padding_word*dilation,
                  dilation=dilation) for dilation in dilation_l
        ])
        self.conv_layers_size = len(self.cov_layers)
        self.dense = nn.Linear(in_features=(cnn_num_filters*blocks),
                               out_features=num_labels)
        self.i_drop = nn.Dropout(input_dropout)
        self.m_drop = nn.Dropout(middle_dropout)

    def forward(self, feature):
        feature = self.token_embedding(feature)
        feature = self.i_drop(feature)
        feature = feature.permute(0, 2, 1)
        conv0 = self.conv0(feature)
        conv0 = F.relu(conv0)
        conv_layer = conv0
        conv_outputs = []
        for _ in range(self.num_blocks):
            for j, mdv in enumerate(self.cov_layers):
                conv_layer = mdv(conv_layer)
                conv_layer = F.relu(conv_layer)
                if j == self.conv_layers_size - 1:
                    conv_layer = self.m_drop(conv_layer)
                    conv_outputs.append(conv_layer)
        layer_concat = torch.cat(conv_outputs, 1)
        layer_concat = layer_concat.permute(0, 2, 1)
        return self.dense(layer_concat)
