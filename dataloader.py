#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/03/20
@description:
"""
import json
import os
from typing import Optional, Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchtext.legacy import data, datasets


class NERDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data/corpus", batch_size=128, experiment=False):
        super().__init__()
        self.data_path = data_dir
        self.batch_size = batch_size
        self.experiment = experiment
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        WORD = data.Field(batch_first=True, include_lengths=True)
        TAG = data.Field(batch_first=True, include_lengths=True)

        train_set, val_set, test_set = datasets.UDPOS.splits(path=self.data_path,
                                                             train='train.char.bmes',
                                                             validation='dev.char.bmes',
                                                             test='test.char.bmes',
                                                             fields=(('word', WORD), ('tag', TAG)),
                                                             separator=' ')
        if self.experiment:
            train_set.examples = train_set.examples[: 1000]
            val_set.examples = val_set.examples[: 1000]
            test_set.examples = test_set.examples[: 1000]

        WORD.build_vocab(train_set.word, val_set.word, test_set.word)
        TAG.build_vocab(train_set.tag, val_set.tag, test_set.tag)
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train_set, val_set, test_set),
            batch_size=self.batch_size,
            sort_within_batch=True,
            shuffle=True
        )
        self.char2idx = WORD.vocab.stoi
        self.id2char = WORD.vocab.itos
        self.tag2idx = TAG.vocab.stoi
        self.idx2tag = {index: value for index, value in enumerate(TAG.vocab.itos)}

        self.tag_size = len(TAG.vocab.stoi)
        self.word_size = len(WORD.vocab.stoi)
        self.vocab_size = self.word_size

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x = batch.word[0]
        y = batch.tag[0]
        real_length = batch.word[1]
        return x, y, real_length

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_iter

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_iter

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_iter

    def save_dict(self, data_dir):
        with open(os.path.join(data_dir, "index2tag.txt"), 'w', encoding='utf8') as writer:
            json.dump(self.idx2tag, writer, ensure_ascii=False)

        with open(os.path.join(data_dir, "token2index.txt"), 'w', encoding='utf8') as writer:
            json.dump(self.char2idx, writer, ensure_ascii=False)
