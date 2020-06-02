import sys
import os
import argparse
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
from tensorboardX import SummaryWriter

import sru
import flop
from flambe.optim import RAdam
from flop.embedding import AdaptiveEmbedding, AdaptiveLogSoftmax
from flop.embedding import HardConcreteAdaptiveEmbedding, HardConcreteAdaptiveLogSoftmax
from utils.data_utils import get_lm_corpus

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        #self.cutoffs = [20000, 60000]
        self.cutoffs = [10000, 20000, 40000, 60000, 100000]
        self.n_V = args.n_token
        self.n_e = args.n_e or args.n_proj
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = AdaptiveEmbedding(self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            div_freq=2,
            dropout=args.dropout_e
        )
        self.rnn = sru.SRU(self.n_d, self.n_d, self.depth,
            projection_size=args.n_proj,
            dropout=args.dropout,
            highway_bias=args.bias,
            layer_norm=args.layer_norm,
            rescale=args.rescale,
            custom_m=flop.ProjectedLinear(
                self.n_d, self.n_d * 3,
                proj_features=args.n_proj,
                bias=False
            )
        )
        self.output_layer = AdaptiveLogSoftmax(self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            div_freq=2,
            dropout=args.dropout_e,
            keep_order=False
        )
        self.init_weights()
        if not args.not_tie:
            self.tie_weights()

    def tie_weights(self):
        for i in range(len(self.output_layer.out_layers)):
            self.embedding_layer.emb_layers[i].weight = self.output_layer.out_layers[i].weight

        for i in range(len(self.output_layer.out_projs)):
            self.embedding_layer.emb_projs[i] = self.output_layer.out_projs[i]

        if hasattr(self.embedding_layer, 'masks') and hasattr(self.output_layer, 'masks'):
            delattr(self.output_layer, 'masks')
            setattr(self.output_layer, 'masks', self.embedding_layer.masks)

    def init_weights(self, init_range=0.03, reinit_rnn=False):
        params = list(self.embedding_layer.parameters()) + list(self.output_layer.parameters())
        for p in params:
            if p.dim() > 1:  # matrix
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.zero_()
        if reinit_rnn:
            for p in self.rnn.parameters():
                if p.dim() > 1:  # matrix
                    p.data.uniform_(-init_range, init_range)

    def forward(self, x, y, hidden):
        emb = self.drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        loss = self.output_layer(output, y.view(-1))
        loss = loss.view(y.size(0), -1)
        return loss, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = weight.new(self.depth, batch_size, self.n_d).zero_()
        return zeros

def calc_norm(lis):
    l2_sum = sum(x.norm()**2 for x in lis)
    return l2_sum**0.5

def eval_model(model, valid):
    with torch.no_grad():
        # Important: reset compiled masks. When multiple GPUs are used, model() and DDP model()
        # are not the same instance although they share the same parameters.
        # Calling model(..) in training mode will reset all compiled weights and cached masks
        for x, y, seq_len in valid:
            model(x, y, hidden=None)
            break
        model.eval()
        args = model.args
        batch_size = args.eval_batch_size or args.batch_size
        total_loss = 0.0
        total_tok = 0.0
        hidden = model.init_hidden(batch_size)
        for x, y, seq_len in valid:
            loss, hidden = model(x, y, hidden)
            total_loss += loss.sum().item()
            total_tok += y.numel()
        avg_loss = total_loss / total_tok
        ppl = np.exp(avg_loss)
        model.train()
        return ppl, avg_loss

def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone().cpu()
    return states

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    if args.local_rank == 0:
        log_path = "{}_{}".format(args.log, random.randint(1,100))
        train_writer = SummaryWriter(log_dir=log_path+"/train")
        dev_writer = SummaryWriter(log_dir=log_path+"/dev")

    # set up distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    set_seed(1234)
    args.n_gpu = 1
    args.device = device
    local_rank = args.local_rank

    corpus = get_lm_corpus(args.data, 'wt103')
    n_token = args.n_token = len(corpus.vocab)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_unroll_size = args.eval_unroll_size or args.unroll_size
    unroll_size = args.unroll_size
    eval_unroll_size = args.eval_unroll_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    n_nodes = torch.cuda.device_count()
    train = corpus.get_distributed_iterator('train', batch_size,
                                            unroll_size, n_nodes=n_nodes,
                                            rank=local_rank, device=device)
    dev = corpus.get_iterator('valid', eval_batch_size, eval_unroll_size, device=device)
    if local_rank == 0:
        print("vocab size: {}".format(n_token))

    model = Model(args)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    lr = 1.0 if not args.noam else 1.0/(args.n_d**0.5)/(args.warmup_steps**1.5)
    if args.prune:
        # in place substituion of linear ops in SRU
        flop.make_hard_concrete(model.rnn, in_place=True, init_mean=args.prune_init_mean)
        model.embedding_layer = HardConcreteAdaptiveEmbedding.from_module(
                model.embedding_layer,
                init_mean=args.prune_init_mean
        )
        model.output_layer = HardConcreteAdaptiveLogSoftmax.from_module(
                model.output_layer,
                init_mean=args.prune_init_mean
        )
        # tie weights again
        model.tie_weights()
        model.to(device)
        hc_modules = flop.get_hardconcrete_modules(model.rnn) + flop.get_hardconcrete_modules(model.embedding_layer)
        #print(len(flop.get_hardconcrete_modules(model)))
        #print(len(hc_modules))
        hc_parameters = [p for m in hc_modules for p in m.parameters() if p.requires_grad]
        optimizer_hc = RAdam(
            hc_parameters,
            lr = lr * args.prune_lr,
            weight_decay = 0
        )

        lambda_1 = nn.Parameter(torch.tensor(0.).cuda())
        lambda_2 = nn.Parameter(torch.tensor(0.).cuda())
        optimizer_max = RAdam(
            [lambda_1, lambda_2],
            lr = lr,
            weight_decay = 0
        )
        optimizer_max.param_groups[0]['lr'] = -lr * args.prune_lr
        hc_linear_modules = flop.get_hardconcrete_linear_modules(model) + \
                [model.embedding_layer]

        num_hardconcrete_params = sum(x.numel() for x in hc_parameters)
        num_prunable_params = sum(m.num_prunable_parameters() for m in hc_linear_modules)
        if local_rank == 0:
            print("num of hardconcrete paramters: {}".format(num_hardconcrete_params))
            print("num of prunable paramters: {}".format(num_prunable_params))
    else:
        model.to(device)
        args.prune_start_epoch = args.max_epoch

    m_parameters = [i[1] for i in model.named_parameters() if i[1].requires_grad and 'log_alpha' not in i[0]]
    optimizer = RAdam(
        m_parameters,
        lr = lr * args.lr,
        weight_decay = args.weight_decay
    )
    num_params = sum(x.numel() for x in m_parameters if x.requires_grad)

    model_ = model
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        dim=1,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    nbatch = 1
    niter = 1
    best_dev = 1e+8
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = train.n_batch
    checkpoint = None
    if local_rank == 0:
        print(model)
        print("num of parameters: {}".format(num_params))
        print("num of mini-batches: {}".format(N))

    model.zero_grad()
    if args.prune:
        optimizer_max.zero_grad()
        optimizer_hc.zero_grad()

    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        hidden = model_.init_hidden(batch_size)
        start_prune = epoch >= args.prune_start_epoch
        i = 0

        for x, y, seq_len in train:
            i += 1
            hidden.detach_()

            # language model forward and backward
            loss, hidden = model(x, y, hidden)
            loss = loss.mean()
            (loss / args.update_param_freq).backward()
            loss = loss.item()
            lagrangian_loss = 0
            target_sparsity = 0
            expected_sparsity = 0

            # add lagrangian loss (regularization) when pruning
            if start_prune:
                # compute target sparsity with (optionally) linear warmup
                target_sparsity = args.prune_sparsity
                if args.prune_warmup > 0:
                    niter_ = niter - args.prune_start_epoch * N
                    target_sparsity *= min(1.0, niter_ / args.prune_warmup)

                # compute expected model size and sparsity
                expected_size = sum(m.num_parameters(train=True) for m in hc_linear_modules)
                expected_sparsity = 1.0 - expected_size / num_prunable_params

                # compute lagrangian loss
                lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + \
                                  lambda_2 * (expected_sparsity - target_sparsity)**2 * args.prune_beta
                (lagrangian_loss / args.update_param_freq).backward()
                expected_sparsity = expected_sparsity.item()
                lagrangian_loss = lagrangian_loss.item()

            #  log training stats
            if local_rank == 0 and (niter - 1) % 100 == 0 and nbatch % args.update_param_freq == 0:
                if args.prune:
                    train_writer.add_scalar('sparsity/expected_sparsity', expected_sparsity, niter)
                    train_writer.add_scalar('sparsity/target_sparsity', target_sparsity, niter)
                    train_writer.add_scalar('loss/lagrangian_loss', lagrangian_loss, niter)
                    train_writer.add_scalar('lambda/1', lambda_1.item(), niter)
                    train_writer.add_scalar('lambda/2', lambda_2.item(), niter)
                    if (nbatch - 1) % 3000 == 0:
                        for index, layer in enumerate(hc_modules):
                            train_writer.add_histogram(
                                'log_alpha/{}'.format(index),
                                layer.log_alpha,
                                niter,
                                bins='sqrt',
                            )
                sys.stderr.write("\r{:.4f} {:.2f} {:.2f} eta={:.1f}m".format(
                    math.exp(loss),
                    lagrangian_loss,
                    expected_sparsity,
                    (time.time()-start_time)/60.0/(i+1)*(N-i-1),
                ))
                train_writer.add_scalar('loss/ppl', math.exp(loss), niter)
                train_writer.add_scalar('loss/lm_loss', loss, niter)
                train_writer.add_scalar('loss/total_loss', loss + lagrangian_loss, niter)
                train_writer.add_scalar('parameter_norm',
                    calc_norm([ x.data for x in m_parameters ]),
                    niter
                )
                train_writer.add_scalar('gradient_norm',
                    calc_norm([ x.grad for x in m_parameters if x.grad is not None]),
                    niter
                )

            #  perform gradient decent every few number of backward()
            if nbatch % args.update_param_freq == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm(m_parameters, args.clip_grad)
                optimizer.step()
                if start_prune:
                    optimizer_max.step()
                    optimizer_hc.step()
                #  clear gradient
                model.zero_grad()
                if args.prune:
                    optimizer_max.zero_grad()
                    optimizer_hc.zero_grad()
                niter += 1

            if local_rank == 0 and (nbatch % args.log_period == 0 or i == N):
                elapsed_time = (time.time()-start_time)/60.0
                dev_ppl, dev_loss = eval_model(model_, dev)
                dev_writer.add_scalar('loss/lm_loss', dev_loss, niter)
                dev_writer.add_scalar('loss/ppl', dev_ppl, niter)
                dev_writer.add_scalar('ppl', dev_ppl, niter)
                sparsity = 0
                if args.prune:
                    pruned_size = sum(m.num_parameters(train=False) for m in hc_linear_modules)
                    sparsity = 1.0 - pruned_size / num_prunable_params
                    dev_writer.add_scalar('sparsity/hard_sparsity', sparsity, niter)
                    dev_writer.add_scalar('model_size/total_prunable', num_prunable_params, niter)
                    dev_writer.add_scalar('model_size/current_prunable', pruned_size, niter)
                    dev_writer.add_scalar('model_size/total', num_params, niter)
                    dev_writer.add_scalar('model_size/current',
                        num_params - num_prunable_params + pruned_size,
                        niter
                    )
                    dev_writer.add_scalar('model_size/current_embedding',
                        model_.embedding_layer.num_parameters(train=False),
                        niter
                    )
                    dev_writer.add_scalar('model_size/current_output_layer',
                        model_.output_layer.num_parameters(train=False),
                        niter
                    )
                sys.stdout.write("\rnum_batches={}  lr={:.5f}  train_loss={:.4f}  dev_loss={:.4f}"
                        "  dev_bpc={:.2f}  sparsity={:.2f}\t[{:.1f}m]\n".format(
                    nbatch,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    dev_loss,
                    dev_ppl,
                    sparsity,
                    elapsed_time
                ))
                if dev_ppl < best_dev:
                    if (not args.prune) or sparsity > args.prune_sparsity - 0.02:
                        best_dev = dev_ppl
                        checkpoint = copy_model(model_)
                sys.stdout.write("\n")
                sys.stdout.flush()

            nbatch += 1
            if args.noam:
                lr = min(1.0 / (niter**0.5), niter / (args.warmup_steps**1.5))
                optimizer.param_groups[0]['lr'] = lr * args.lr / (args.n_d**0.5)
            if args.noam and start_prune:
                niter_ = niter - args.prune_start_epoch * N
                lr = min(1.0 / (niter_**0.5), niter_ / (args.warmup_steps**1.5))
                optimizer_max.param_groups[0]['lr'] = -lr * args.prune_lr / (args.n_d**0.5)
                optimizer_hc.param_groups[0]['lr'] = lr * args.lr / (args.n_d**0.5)

        if local_rank == 0 and args.save and (epoch + 1) % 10 == 0:
            torch.save(copy_model(model_), "{}.{}.{}.pt".format(
                args.save,
                epoch + 1,
                int(dev_ppl)
                #sparsity
            ))

    if local_rank == 0:
        train_writer.close()
        dev_writer.close()

        if checkpoint is not None:
            model_.load_state_dict(checkpoint)
            model_.to(device)
        #dev = create_batches(dev_, 1)
        #test = create_batches(test_, 1)
        test = corpus.get_iterator('test', eval_batch_size, eval_unroll_size, device=device)
        dev_ppl, dev_loss = eval_model(model_, dev)
        test_ppl, test_loss = eval_model(model_, test)
        sys.stdout.write("dev_ppl={:.3f}  test_ppl={:.3f}\n".format(
            dev_ppl, test_ppl
        ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--log", type=str, required=True)
    argparser.add_argument("--noam", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=4000)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--rescale", action="store_true")
    argparser.add_argument("--not_tie", action="store_true")
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--update_param_freq", type=int, default=1)
    argparser.add_argument("--batch_size", "--batch", type=int, default=24)
    argparser.add_argument("--eval_batch_size", type=int, default=10)
    argparser.add_argument("--unroll_size", type=int, default=256)
    argparser.add_argument("--eval_unroll_size", type=int, default=0)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--n_e", type=int, default=1024)
    argparser.add_argument("--n_d", "--d", type=int, default=2048)
    argparser.add_argument("--n_proj", type=int, default=512)
    argparser.add_argument("--div_val", type=float, default=4)
    argparser.add_argument("--dropout", type=float, default=0.1,
        help="dropout probability"
    )
    argparser.add_argument("--dropout_e", type=float, default=0.1)
    argparser.add_argument("--bias", type=float, default=-3,
        help="intial bias of highway gates",
    )
    argparser.add_argument("--depth", type=int, default=12)
    argparser.add_argument("--lr", type=float, default=2)
    argparser.add_argument("--weight_decay", type=float, default=0)
    argparser.add_argument("--clip_grad", type=float, default=0.3)
    argparser.add_argument("--log_period", type=int, default=1000000)
    argparser.add_argument("--save", type=str, default="")
    argparser.add_argument("--load", type=str, default="")

    argparser.add_argument("--prune", action="store_true")
    argparser.add_argument("--prune_lr", type=float, default=3)
    argparser.add_argument("--prune_beta", type=float, default=1)
    argparser.add_argument("--prune_warmup", type=int, default=0)
    argparser.add_argument("--prune_sparsity", type=float, default=0.)
    argparser.add_argument("--prune_init_mean", type=float, default=0.05)
    argparser.add_argument("--prune_start_epoch", type=int, default=0)

    argparser.add_argument("--local_rank", type=int, default=0)
    args = argparser.parse_args()
    print (args)
    main(args)
