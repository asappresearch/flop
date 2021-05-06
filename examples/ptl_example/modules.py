import sys
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import sru
import flop


class Enwik8Dataset(Dataset):
    def __init__(self,
                 source: torch.Tensor,
                 target: torch.Tensor,
                 unroll_size: int):
        super(Enwik8Dataset, self).__init__()
        self.unroll_size = unroll_size
        self._len = (source.size(0) - 1) // unroll_size + 1
        self._source = source
        self._target = target

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        unroll_size = self.unroll_size
        return self._source[idx * unroll_size:(idx + 1) * unroll_size], \
            self._target[idx * unroll_size:(idx + 1) * unroll_size]


class Enwik8DataModule(LightningDataModule):
    def __init__(self,
                 path: str,
                 unroll_size: int,
                 batch_size: int,
                 valid_batch_size: int = 0,
                 test_batch_size: int = 0,
                 num_test_symbols: int = 5000000):
        super(Enwik8DataModule, self).__init__()

        self.path = path
        self.unroll_size = unroll_size
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_test_symbols = num_test_symbols

    def setup(self, stage: Optional[str] = None):
        sys.stdout.write("setup(stage={})".format(stage))
        raw_data = open(self.path).read()
        raw_data = np.fromstring(raw_data, dtype=np.uint8)
        unique, data = np.unique(raw_data, return_inverse=True)
        num_test_symbols = self.num_test_symbols
        self._train_data = data[: -2 * num_test_symbols]
        self._valid_data = data[-2 * num_test_symbols: -num_test_symbols]
        self._test_data = data[-num_test_symbols:]
        self._unique = unique

    def create_dataset(self,
                       data_ids,
                       batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(data_ids)
        L = ((N-1) // batch_size) * batch_size
        x = np.copy(data_ids[:L].reshape(batch_size, -1).T)
        y = np.copy(data_ids[1:L+1].reshape(batch_size, -1).T)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.contiguous(), y.contiguous()
        return Enwik8Dataset(x, y, self.unroll_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.create_dataset(self._train_data, self.batch_size),
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.create_dataset(self._valid_data, self.valid_batch_size),
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.create_dataset(self._test_data, self.test_batch_size),
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )


class SRUModel(nn.Module):
    def __init__(self, args: Namespace):
        super(SRUModel, self).__init__()
        self.args = args
        self.n_e = args.n_e or args.vocab_size
        self.n_d = args.n_d
        self.depth = args.depth
        self.vocab_size = args.vocab_size

        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.n_e)
        custom_m_list = [nn.Linear(self.n_e, self.n_d * 4, bias=False)]
        for i in range(self.depth-1):
            if args.n_proj > 0:
                custom_m_list.append(flop.ProjectedLinear(
                    self.n_d, self.n_d * 3,
                    proj_features=args.n_proj,
                    bias=False
                ))
            else:
                custom_m_list.append(nn.Linear(
                    self.n_d, self.n_d * 3,
                    bias=False
                ))
        self.rnn = sru.SRU(
            self.n_e, self.n_d, self.depth,
            dropout=args.dropout,
            highway_bias=args.bias,
            layer_norm=args.layer_norm,
            rescale=args.rescale,
            custom_m=custom_m_list
        )
        self.output_layer = nn.Linear(self.n_d, self.vocab_size)
        self.init_weights()

    def init_weights(self, reinit_rnn=False):
        params = list(self.embedding_layer.parameters()) + \
            list(self.output_layer.parameters())
        for p in params:
            if p.dim() > 1:  # matrix
                val = (3.0/p.size(0))**0.5
                p.data.uniform_(-val, val)
            else:
                p.data.zero_()
        if reinit_rnn:
            for p in self.rnn.parameters():
                if p.dim() > 1:  # matrix
                    val = (3.0/p.size(0))**0.5
                    p.data.uniform_(-val, val)

    def forward(self, x, hidden):
        emb = self.drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = weight.new(self.depth, batch_size, self.n_d).zero_()
        return zeros
