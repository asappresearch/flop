from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardconcrete import HardConcrete

class AdaptiveEmbedding(nn.Module):
'''
    Code taken and modified from Transformer-XL
    https://github.com/kimiyoung/transformer-xl
'''
    def __init__(self, n_token, d_embed, d_proj, cutoffs,
                 div_val=1,
                 div_freq=1,
                 dropout=0.0):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.div_freq = div_freq
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.dropout = nn.Dropout(p=dropout)
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
            d_emb_i = int(d_embed // (div_val ** (i // div_freq)))
            self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):

        embeddings = [l.weight for l in self.emb_layers]
        emb_projs = self.emb_projs

        return self._forward(inp, embeddings, emb_projs)

    def _forward(self, inp, embeddings, emb_projs):

        param = next(self.parameters())
        inp_flat = inp.view(-1)
        emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
            dtype=param.dtype, device=param.device)

        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

            mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            inp_i = inp_flat.index_select(0, indices_i) - l_idx
            emb_i = F.embedding(inp_i, embeddings[i], None, None, 2., False, False)
            emb_i = F.linear(self.dropout(emb_i), emb_projs[i])

            emb_flat.index_copy_(0, indices_i, emb_i)

        embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class HardConcreteAdaptiveEmbedding(AdaptiveEmbedding):
    def __init__(self, n_token, d_embed, d_proj, cutoffs,
                 div_val=1,
                 div_freq=1,
                 dropout=0.0,
                 init_mean: float = 0.5,
                 init_std: float = 0.01):

        super(HardConcreteAdaptiveEmbedding, self).__init__(
            n_token, d_embed, d_proj, cutoffs,
            div_val=div_val,
            div_freq=div_freq,
            dropout=dropout
        )

        self.masks = nn.ModuleList()
        for i in range(len(self.cutoffs)):
            d_emb_i = self.emb_projs[i].size(1)
            self.masks.append(HardConcrete(d_emb_i, init_mean, init_std))

        self.indices = None
        self.compiled_projs = None
        self.compiled_embeddings = None

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return sum(l.weight.numel() for l in self.emb_layers) + \
               sum(weight.numel() for weight in self.emb_projs)

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        params = torch.tensor(0, dtype=torch.float).to(self.emb_projs[0])
        if train:
            for i in range(len(self.cutoffs)):
                n_proj = self.masks[i].l0_norm()
                params += (self.emb_projs[i].size(0) + self.emb_layers[i].weight.size(0)) * n_proj
        elif self.compiled_projs is not None and self.compiled_embeddings is not None:
            for i in range(len(self.cutoffs)):
                if len(self.indices[i]) == 0:
                    warnings.warn("Mask is all zero in layer-{} AdaptiveEmbedding".format(i), RuntimeWarning)
                else:
                    params += self.compiled_projs[i].numel() + \
                              self.compiled_embeddings[i].numel()
        return params

    def forward(self, inp, **kwargs):

        embeddings = None
        emb_projs = None

        if self.training:
            # Reset masks and compiled weights
            self.compiled_projs = None
            self.compiled_embeddings = None
            self.indices = None
        elif self.compiled_projs is not None:
            embeddings = self.compiled_embeddings
            emb_projs = self.compiled_projs

        if embeddings is None:
            indices = []
            embeddings = []
            emb_projs = []

            # Sample mask and compute weights
            for i in range(len(self.cutoffs)):
                mask_i = self.masks[i]()
                indices_i = mask_i.data.nonzero().view(-1)
                dim_i = self.emb_projs[i].size(1)

                if len(indices_i) == 0:
                    warnings.warn("Mask is all zero in AdaptiveEmbedding layer-{}".format(i), RuntimeWarning)

                if len(indices_i) == 0 or len(indices_i) > dim_i * 0.8:
                    embedding_i = self.emb_layers[i].weight * mask_i.view(1, -1)
                    emb_proj_i = self.emb_projs[i]
                else:
                    embedding_i = self.emb_layers[i].weight * mask_i.view(1, -1)
                    embedding_i = embedding_i.index_select(1, indices_i)
                    emb_proj_i = self.emb_projs[i].index_select(1, indices_i)

                indices.append(indices_i)
                embeddings.append(embedding_i)
                emb_projs.append(emb_proj_i)

            if not self.training:
                self.indices = indices
                self.compiled_embeddings = embeddings
                self.compiled_projs = emb_projs
                # debug
                #print("\n", [len(indices_i) for indices_i in self.indices])

        return self._forward(inp, embeddings, emb_projs)

    @classmethod
    def from_module(cls,
                    module: AdaptiveEmbedding,
                    init_mean: float = 0.5,
                    init_std: float = 0.01,
                    keep_weights: bool = True) -> 'HardConcreteAdaptiveEmbedding':

        n_token = module.n_token
        d_embed = module.d_embed
        d_proj = module.d_proj
        cutoffs = module.cutoffs[:-1]
        div_val = module.div_val
        div_freq = module.div_freq
        dropout = module.dropout.p

        new_module = cls(n_token, d_embed, d_proj, cutoffs,
                         div_val=div_val,
                         div_freq=div_freq,
                         dropout=dropout,
                         init_mean=init_mean,
                         init_std=init_std)

        if keep_weights:
            for i in range(len(module.cutoffs)):
                new_module.emb_projs[i].data = module.emb_projs[i].data.clone()
                new_module.emb_layers[i].weight.data = module.emb_layers[i].weight.data.clone()

        return new_module


class AdaptiveLogSoftmax(nn.Module):
'''
    Code taken and modified from Transformer-XL
    https://github.com/kimiyoung/transformer-xl
'''
    def __init__(self, n_token, d_embed, d_proj, cutoffs,
                 div_val=1,
                 div_freq=1,
                 dropout=0.0, keep_order=True):
        super(AdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.div_freq = div_freq

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.dropout = nn.Dropout(p=dropout)
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
            d_emb_i = int(d_embed // (div_val ** (i // div_freq)))
            output_size_i = r_idx - l_idx if i > 0 else (r_idx - l_idx) + self.n_clusters

            self.out_projs.append(
                nn.Parameter(torch.Tensor(d_proj, d_emb_i))
            )
            self.out_layers.append(nn.Linear(d_emb_i, output_size_i))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(self.dropout(proj_hid), weight, bias=bias)

        return logit

    def forward(self, hidden, target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        # construct weights and biases
        weights = [l.weight for l in self.out_layers]
        biases = [l.bias for l in self.out_layers]
        out_projs = self.out_projs

        return self._forward(hidden, target,
                             weights, biases, out_projs,
                             keep_order=keep_order)

    def _forward(self, hidden, target,
                 weights, biases, out_projs,
                 keep_order=False):

        head_weight, head_bias, head_proj = weights[0], biases[0], out_projs[0]

        head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
        head_logprob = F.log_softmax(head_logit, dim=1)

        if self.n_clusters == 0:
            return -head_logprob.gather(1, target.unsqueeze(1)).squeeze(1)

        nll = torch.zeros_like(target,
                dtype=hidden.dtype, device=hidden.device)

        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

            mask_i = (target >= l_idx) & (target < r_idx)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
            else:
                weight_i, bias_i, proj_i = weights[i], biases[i], out_projs[i]

                hidden_i = hidden.index_select(0, indices_i)

                tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                logprob_i = head_logprob_i[:, -i] \
                          + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

            if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

            offset += logprob_i.size(0)

        return nll


class HardConcreteAdaptiveLogSoftmax(AdaptiveLogSoftmax):
    def __init__(self, n_token, d_embed, d_proj, cutoffs,
                 div_val=1,
                 div_freq=1,
                 dropout=0.0,
                 keep_order=True,
                 init_mean: float = 0.5,
                 init_std: float = 0.01):

        super(HardConcreteAdaptiveLogSoftmax, self).__init__(
            n_token, d_embed, d_proj, cutoffs,
            div_val=div_val,
            div_freq=div_freq,
            dropout=dropout,
            keep_order=keep_order
        )

        self.masks = nn.ModuleList()
        for i in range(len(self.cutoffs)):
            d_emb_i = self.out_projs[i].size(1)
            self.masks.append(HardConcrete(d_emb_i, init_mean, init_std))

        self.indices = None
        self.compiled_projs = None
        self.compiled_embeddings = None

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return sum(l.weight.numel() for l in self.out_layers) + \
               sum(weight.numel() for weight in self.out_projs)

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        params = torch.tensor(0, dtype=torch.float).to(self.out_projs[0])
        if train:
            for i in range(len(self.cutoffs)):
                n_proj = self.masks[i].l0_norm()
                params += (self.out_projs[i].size(0) + self.out_layers[i].weight.size(0)) * n_proj
        elif self.compiled_projs is not None and self.compiled_embeddings is not None:
            for i in range(len(self.cutoffs)):
                if len(self.indices[i]) == 0:
                    warnings.warn("Mask is all zero in AdaptiveSoftmax layer-{}".format(i), RuntimeWarning)
                else:
                    params += self.compiled_projs[i].numel() + \
                              self.compiled_embeddings[i].numel()
        return params

    def forward(self, hidden, target, keep_order=False, **kwargs):

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        # construct weights and biases
        biases = [l.bias for l in self.out_layers]
        weights = None
        out_projs = None

        if self.training:
            # Reset masks and compiled weights
            self.compiled_projs = None
            self.compiled_embeddings = None
            self.indices = None
        elif self.compiled_projs is not None:
            weights = self.compiled_embeddings
            out_projs = self.compiled_projs

        if weights is None:
            indices = []
            weights = []
            out_projs = []

            # Sample mask and compute weights
            for i in range(len(self.cutoffs)):
                mask_i = self.masks[i]()
                indices_i = mask_i.data.nonzero().view(-1)
                dim_i = self.out_projs[i].size(0)

                if len(indices_i) == 0:
                    warnings.warn("Mask is all zero in AdaptiveSoftmax layer-{}".format(i), RuntimeWarning)

                if len(indices_i) == 0 or len(indices_i) > dim_i * 0.8:
                    embedding_i = self.out_layers[i].weight * mask_i.view(1, -1)
                    emb_proj_i = self.out_projs[i]
                else:
                    embedding_i = self.out_layers[i].weight * mask_i.view(1, -1)
                    embedding_i = embedding_i.index_select(1, indices_i)
                    emb_proj_i = self.out_projs[i].index_select(1, indices_i)

                indices.append(indices_i)
                weights.append(embedding_i)
                out_projs.append(emb_proj_i)

            if not self.training:
                self.indices = indices
                self.compiled_embeddings = weights
                self.compiled_projs = out_projs
                # debug
                #print("\n", [len(indices_i) for indices_i in self.indices])

        return self._forward(hidden, target,
                             weights, biases, out_projs,
                             keep_order=keep_order)

    @classmethod
    def from_module(cls,
                    module: AdaptiveLogSoftmax,
                    init_mean: float = 0.5,
                    init_std: float = 0.01,
                    keep_weights: bool = True) -> 'HardConcreteAdaptiveLogSoftmax':

        n_token = module.n_token
        d_embed = module.d_embed
        d_proj = module.d_proj
        cutoffs = module.cutoffs[:-1]
        div_val = module.div_val
        div_freq = module.div_freq
        dropout = module.dropout.p
        keep_order = module.keep_order

        new_module = cls(n_token, d_embed, d_proj, cutoffs,
                         div_val=div_val,
                         div_freq=div_freq,
                         dropout=dropout,
                         keep_order=keep_order,
                         init_mean=init_mean,
                         init_std=init_std)

        if keep_weights:
            for i in range(len(module.cutoffs)):
                new_module.out_projs[i].data = module.out_projs[i].data.clone()
                new_module.out_layers[i].weight.data = module.out_layers[i].weight.data.clone()
                new_module.out_layers[i].bias.data = module.out_layers[i].bias.data.clone()

        return new_module
