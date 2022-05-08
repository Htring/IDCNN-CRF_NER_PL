#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2021/11/21
@description:
"""
import json
import os
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from model.idcnn_crf_pl import IDCNN_CRF
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from dataloader import NERDataModule
from model.idcnn import IDCNN

pl.seed_everything(2022)


def train(args):
    path_prefix = "model_save"
    os.makedirs(path_prefix, exist_ok=True)

    ner_dm = NERDataModule(data_dir=args.data_path, batch_size=args.batch_size)
    args.tag_size = ner_dm.tag_size
    args.vocab_size = ner_dm.vocab_size
    args.id2char = ner_dm.id2char
    args.idx2tag = ner_dm.idx2tag
    if args.load_pre:
        model = IDCNN_CRF.load_from_checkpoint(args.ckpt_path, hparams=args)
    else:
        model = IDCNN_CRF(args)
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                          monitor="val_f1",
                                          mode="max",
                                          dirpath=path_prefix,
                                          filename="ner-{epoch:03d}-{val_f1:.3f}", )
    trainer = Trainer.from_argparse_args(args, callbacks=[lr_logger,
                                                          checkpoint_callback],
                                         gpus=1,
                                         max_epochs=500)

    if args.train:
        trainer.fit(model=model, datamodule=ner_dm)

    if args.test:
        trainer.test(model, ner_dm)

    if args.save_state_dict:
        if len(os.name) > 0:
            ner_dm.save_dict(path_prefix)


def model_use(param):
    model_dir = os.path.dirname(param.ckpt_path)

    def _load_dict():
        with open(os.path.join(model_dir, "token2index.txt"), 'r', encoding='utf8') as reader:
            t2i_dict: dict = json.load(reader)
        t2i_dict = {token: int(index) for token, index in t2i_dict.items()}
        with open(os.path.join(model_dir, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            i2t_dict: dict = json.load(reader)
        i2t_dict = {int(index): tag for index, tag in i2t_dict.items()}
        return t2i_dict, i2t_dict

    def num_data(content: str, token2index: dict):
        number_data = [token2index.get(char, token2index.get("<unk>")) for char in content]
        return number_data

    token2index, index2tag = _load_dict()
    param.tag_size = len(index2tag)
    param.vocab_size = len(token2index)
    param.idx2tag = index2tag
    param.id2char = {index: char for index, char in enumerate(token2index.keys())}
    model = IDCNN_CRF.load_from_checkpoint(param.ckpt_path, hparams=param)

    test_data = "常建良，男，"
    # encode
    input_data = torch.tensor([num_data(test_data, token2index)], dtype=torch.long)
    # predict
    predict = model(input_data)[0]
    result = []
    # decode
    for predict_id in predict:
        result.append(index2tag.get(predict_id.item()))
    print(predict)
    print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--load_pre", default=True, action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="model_save/ner-epoch=151-val_f1=0.934.ckpt")
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--save_state_dict", default=True, action="store_true")
    parser = IDCNN.add_model_specific_args(parser)
    params = parser.parse_args()
    # train(params)
    model_use(params)
