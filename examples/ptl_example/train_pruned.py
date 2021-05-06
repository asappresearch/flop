import sys
import argparse
from argparse import Namespace

import torch
import flop

from train_base import Enwik8LightningModule, Enwik8Runner


class Enwik8PruningModule(flop.training.FlopPruningModule, Enwik8LightningModule):
    def __init__(self,
                 args: Namespace):

        super(Enwik8PruningModule, self).__init__(args)
        # initialize unpruned model from a checkpoint
        self.model.load_state_dict(torch.load(args.load_checkpoint))
        # convert model by making submodules into prunable modules
        self.setup_model_for_compression()

    def finalize_compression(self):
        """Demonstrate custom compression such as pruning nn.Embeddings."""
        flop.make_compressed_module(
            self.model,
            in_place=True,
        )
        emb = self.model.embedding_layer
        sru_rnn = self.model.rnn
        first_cell = sru_rnn.rnn_lst[0]
        if isinstance(first_cell.custom_m, flop.ColumnSparseLinear):
            # first rnn layer only needs a subset of embedding dims
            indices = first_cell.custom_m.indices
            emb.weight.data = emb.weight.data.index_select(-1, indices)
            emb.embedding_dim = indices.numel()
            first_cell.custom_m.indices = None


class Enwik8PruningRunner(Enwik8Runner):
    def setup_model_module(self):
        lit_module = Enwik8PruningModule(self.args)
        print(lit_module)
        return lit_module

    def run(self):
        super().run()
        print("\nModel before final compression:")
        lit_module = self.lit_module
        print(lit_module.model)

        print("\nModel after final compression:")
        lit_module.finalize_compression()
        print(lit_module.model)

        print("\nTest result after final compression:")
        trainer = self.trainer
        # need to set model=lit_module otherwise trainer will load from a ckpt.
        trainer.test(model=lit_module, datamodule=self.data_module, verbose=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--project", type=str, default="FLOP Enwik8 (DEBUG)")
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--run_name", type=str, required=True)
    argparser.add_argument("--load_checkpoint", type=str, required=True)
    argparser.add_argument("--noam", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=8000)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--rescale", action="store_true")
    argparser.add_argument("--data", type=str, required=True, help="data file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=64)
    argparser.add_argument("--update_param_freq", type=int, default=1)
    argparser.add_argument("--unroll_size", type=int, default=256)
    argparser.add_argument("--max_epoch", type=int, default=30)
    argparser.add_argument("--n_e", type=int, default=0)
    argparser.add_argument("--n_d", "--d", type=int, default=3056)
    argparser.add_argument("--n_proj", type=int, default=512)
    argparser.add_argument("--dropout", type=float, default=0.2,
                           help="dropout probability")
    argparser.add_argument("--bias", type=float, default=-3,
                           help="intial bias of highway gates")
    argparser.add_argument("--depth", type=int, default=6)
    argparser.add_argument("--lr", type=float, default=2)
    argparser.add_argument("--weight_decay", type=float, default=1e-7)
    argparser.add_argument("--clip_grad", type=float, default=0.3)
    argparser.add_argument("--log_period", type=int, default=50)

    args = argparser.parse_args()
    print(args)
    print(Enwik8PruningModule.mro())

    runner = Enwik8PruningRunner(args)
    runner.run()
