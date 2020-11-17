import sys
import argparse
import time
import random
import math
import os
import zipfile

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import sru
import flop
from flop.embedding import AdaptiveEmbedding, AdaptiveLogSoftmax
from flop.scripts.wt103.utils.data_utils import get_lm_corpus


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        # self.cutoffs = [20000, 60000]
        self.cutoffs = [10000, 20000, 40000, 60000, 100000]
        self.n_V = args.n_token
        self.n_e = args.n_e or args.n_proj
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = AdaptiveEmbedding(
            self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            div_freq=2,
            dropout=args.dropout_e,
        )
        self.rnn = sru.SRU(
            self.n_d,
            self.n_d,
            self.depth,
            projection_size=args.n_proj,
            dropout=args.dropout,
            highway_bias=args.bias,
            layer_norm=args.layer_norm,
            rescale=args.rescale,
            custom_m=flop.ProjectedLinear(
                self.n_d, self.n_d * 3, proj_features=args.n_proj, bias=False
            ),
        )
        self.output_layer = AdaptiveLogSoftmax(
            self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            div_freq=2,
            dropout=args.dropout_e,
            keep_order=False,
        )
        self.init_weights()
        if not args.not_tie:
            self.tie_weights()

    def tie_weights(self):
        for i in range(len(self.output_layer.out_layers)):
            self.embedding_layer.emb_layers[i].weight = self.output_layer.out_layers[
                i
            ].weight

        for i in range(len(self.output_layer.out_projs)):
            self.embedding_layer.emb_projs[i] = self.output_layer.out_projs[i]

        if hasattr(self.embedding_layer, "masks") and hasattr(
            self.output_layer, "masks"
        ):
            delattr(self.output_layer, "masks")
            setattr(self.output_layer, "masks", self.embedding_layer.masks)

    def init_weights(self, init_range=0.03, reinit_rnn=False):
        params = list(self.embedding_layer.parameters()) + list(
            self.output_layer.parameters()
        )
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
    l2_sum = sum(x.norm() ** 2 for x in lis)
    return l2_sum ** 0.5


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
        log_path = "{}_{}".format(args.log, random.randint(1, 100))
        train_writer = SummaryWriter(log_dir=log_path + "/train")
        dev_writer = SummaryWriter(log_dir=log_path + "/dev")

    # set up distributed training
    # torch.cuda.set_device(args.local_rank)
    device = 'cuda'
    # torch.distributed.init_process_group(backend="nccl")
    set_seed(1234)
    args.n_gpu = 1
    args.device = device
    local_rank = args.local_rank

    corpus = get_lm_corpus(args.data, "wt103")
    n_token = args.n_token = len(corpus.vocab)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_unroll_size = args.eval_unroll_size or args.unroll_size
    unroll_size = args.unroll_size
    eval_unroll_size = args.eval_unroll_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    # n_nodes = torch.cuda.device_count()
    # train = corpus.get_distributed_iterator('train', batch_size,
    #                                         unroll_size, n_nodes=n_nodes,
    #                                         rank=local_rank, device=device)
    train = corpus.get_iterator("train", batch_size, unroll_size, device=device)
    dev = corpus.get_iterator("test", eval_batch_size, eval_unroll_size, device=device)
    if local_rank == 0:
        print("vocab size: {}".format(n_token))

    model = Model(args)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    lr = 1.0 if not args.noam else 1.0 / (args.n_d ** 0.5) / (args.warmup_steps ** 1.5)
    if args.prune:
        # tie weights again
        model.tie_weights()
        model.to(device)
        num_mask_params = sum(x.numel() for x in model.parameters())
        num_prunable_params = num_mask_params
        if local_rank == 0:
            print("num of mask parameters: {}".format(num_mask_params))
            print("num of prunable parameters: {}".format(num_prunable_params))

        mask_param_names = [
            i[0]
            for i in model.named_parameters()
            if i[1].requires_grad
        ]
        pruner = flop.NervanaPruner(
            model,
            subpruners={
                "agppruner": {
                    "class": "AutomatedGradualPruner",
                    "initial_sparsity": 0.05,
                    "weights": mask_param_names,
                    "final_sparsity": args.prune_sparsity,
                    "starting_step": args.prune_start_epoch,
                    "ending_step": args.prune_end_epoch,
                    "frequency": 1,
                }
            },
        )
    else:
        model.to(device)
        args.prune_start_epoch = args.max_epoch

    m_parameters = [
        i[1]
        for i in model.named_parameters()
        if i[1].requires_grad
    ]
    optimizer = torch.optim.Adam(m_parameters, lr=lr * args.lr, weight_decay=args.weight_decay)
    num_params = sum(x.numel() for x in m_parameters if x.requires_grad)

    model_ = model
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, dim=1, device_ids=[local_rank], output_device=local_rank,
    # )
    model = nn.DataParallel(model, dim=1).to('cuda')
    nbatch = 1
    niter = 1
    best_dev = 1e8
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = train.n_batch
    checkpoint = None
    if local_rank == 0:
        print(model)
        print("num of parameters: {}".format(num_params))
        print("num of mini-batches: {}".format(N))

    model.zero_grad()

    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        hidden = model_.init_hidden(batch_size)
        i = 0

        pruner.begin_step(epoch)

        for x, y, seq_len in train:
            # start iter on the first batch
            if nbatch % args.update_param_freq == 1:
                pruner.begin_iter(epoch, niter, N // args.update_param_freq)

            i += 1
            hidden.detach_()

            # language model forward and backward
            model_loss, hidden = model(x, y, hidden)

            model_loss = model_loss.mean()
            loss = model_loss

            (loss / args.update_param_freq).backward()
            model_loss = model_loss.item()

            #  log training stats
            if (
                local_rank == 0
                and (niter - 1) % 100 == 0
                and nbatch % args.update_param_freq == 0
            ):
                sys.stderr.write(
                    "\r{:.4f} eta={:.1f}m".format(
                        math.exp(model_loss),
                        (time.time() - start_time) / 60.0 / (i + 1) * (N - i - 1),
                    )
                )
                train_writer.add_scalar("loss/ppl", math.exp(model_loss), niter)
                train_writer.add_scalar(
                    "loss/total_loss", model_loss, niter
                )
                train_writer.add_scalar(
                    "parameter_norm", calc_norm([x.data for x in m_parameters]), niter
                )
                train_writer.add_scalar(
                    "gradieånt_norm",
                    calc_norm([x.grad for x in m_parameters if x.grad is not None]),
                    niter,
                )

            #  perform gradient decent every few number of backward()
            if nbatch % args.update_param_freq == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm(m_parameters, args.clip_grad)
                optimizer.step()
                #  clear gradient
                model.zero_grad()

                # End iter on the last batch
                pruner.end_iter(epoch, niter, N // args.update_param_freq)
                niter += 1

            if local_rank == 0 and (nbatch % args.log_period == 0 or i == N):
                elapsed_time = (time.time() - start_time) / 60.0
                dev_ppl, dev_loss = eval_model(model_, dev)
                dev_writer.add_scalar("loss/lm_loss", dev_loss, niter)
                dev_writer.add_scalar("loss/ppl", dev_ppl, niter)
                dev_writer.add_scalar("ppl", dev_ppl, niter)
                sparsity = 0
                if args.prune:
                    agp_sparsity = pruner.get_step_logs()['sparsity']
                    dev_writer.add_scalar("sparsity/hard_sparsity", agp_sparsity, niter)
                    dev_writer.add_scalar(
                        "model_size/total_prunable", num_prunable_params, niter
                    )
                    dev_writer.add_scalar("model_size/total", num_params, niter)
                sys.stdout.write(
                    "\rnum_batches={}  lr={:.5f}  train_loss={:.4f}  dev_loss={:.4f}"
                    "  dev_bpc={:.2f}  sparsity={:.2f}\t[{:.1f}m]\n".format(
                        nbatch,
                        optimizer.param_groups[0]["lr"],
                        loss,
                        dev_loss,
                        dev_ppl,
                        sparsity,
                        elapsed_time,
                    )
                )
                checkpoint = copy_model(model_)
                sys.stdout.write("\n")
                sys.stdout.flush()

            nbatch += 1
            if args.noam:
                lr = min(1.0 / (niter ** 0.5), niter / (args.warmup_steps ** 1.5))
                optimizer.param_groups[0]["lr"] = lr * args.lr / (args.n_d ** 0.5)

        pruner.end_step(epoch)
        if local_rank == 0 and args.save and (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                "{}.{}.{}.pt".format(
                    args.save,
                    epoch + 1,
                    int(dev_ppl)
                    # sparsity
                ),
            )

    if local_rank == 0:
        train_writer.close()
        dev_writer.close()

        if checkpoint is not None:
            model_.load_state_dict(checkpoint)
            model_.to(device)
        # dev = create_batches(dev_, 1)
        # test = create_batches(test_, 1)
        test = corpus.get_iterator(
            "test", eval_batch_size, eval_unroll_size, device=device
        )
        dev_ppl, dev_loss = eval_model(model_, dev)
        test_ppl, test_loss = eval_model(model_, test)
        sys.stdout.write("dev_ppl={:.3f}  test_ppl={:.3f}\n".format(dev_ppl, test_ppl))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    argparser.add_argument("--log", type=str, default="")
    argparser.add_argument("--noam", type=bool, default=True)
    argparser.add_argument("--warmup_steps", type=int, default=4000)
    argparser.add_argument("--layer_norm", type=bool, default=True)
    argparser.add_argument("--rescale", action="store_true")
    argparser.add_argument("--not_tie", action="store_true")
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--update_param_freq", type=int, default=2)
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--eval_batch_size", type=int, default=10)
    argparser.add_argument("--unroll_size", type=int, default=256)
    argparser.add_argument("--eval_unroll_size", type=int, default=0)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--n_e", type=int, default=1024)
    argparser.add_argument("--n_d", "--d", type=int, default=2048)
    argparser.add_argument("--n_proj", type=int, default=512)
    argparser.add_argument("--div_val", type=float, default=4)
    argparser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout probability"
    )
    argparser.add_argument("--dropout_e", type=float, default=0.1)
    argparser.add_argument(
        "--bias", type=float, default=-3, help="intial bias of highway gates",
    )
    argparser.add_argument("--depth", type=int, default=12)
    argparser.add_argument("--lr", type=float, default=2)
    argparser.add_argument("--weight_decay", type=float, default=0.01)
    argparser.add_argument("--clip_grad", type=float, default=0.3)
    argparser.add_argument("--log_period", type=int, default=1000000)
    argparser.add_argument("--save", type=str, default="")
    argparser.add_argument("--load", type=str, default="")

    argparser.add_argument("--prune", type=bool, default=True)
    argparser.add_argument("--prune_lr", type=float, default=2)
    argparser.add_argument("--prune_warmup", type=int, default=0)
    argparser.add_argument("--prune_start_epoch", type=int, default=0)
    argparser.add_argument("--prune_sparsity", type=float, default=0.8)
    argparser.add_argument("--prune_end_epoch", type=int, default=30)
    argparser.add_argument("--l1_lambda", type=float, default=0)

    argparser.add_argument("--local_rank", type=int, default=0)
    args = argparser.parse_args()

    dirname = os.path.dirname(args.data)
    with zipfile.ZipFile(args.data, 'r') as f:
        f.extractall(dirname)
    args.data = os.path.join(dirname, 'wikitext-103')
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(args.save, exist_ok=True)
    print(args)
    main(args)
