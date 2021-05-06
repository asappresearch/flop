import sys
import argparse
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import wuwei

from modules import SRUModel, Enwik8DataModule


def get_noam_lr_scheduler(args):
    warmup_steps = args.warmup_steps
    scaling = args.n_d ** -0.5

    def lr_lambda(step):
        if step <= warmup_steps:
            return max(0, scaling * step / warmup_steps**1.5)
        else:
            return max(0, scaling / step**0.5)
    return lr_lambda


class Enwik8LightningModule(LightningModule):
    def __init__(self,
                 args: Namespace):

        super(Enwik8LightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = SRUModel(args)
        self.args = args
        self.prev_hidden = None
        self.train_loss_fn = nn.CrossEntropyLoss()
        self.eval_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # track best model on our own
        self.best_bpc = 1e+8
        self.best_checkpoint = None

    def _save_checkpoint(self):
        states = self.model.state_dict()
        for k in states:
            v = states[k]
            states[k] = v.clone().cpu()
        self.best_checkpoint = states

    def _compute_batch(self, batch, loss_fn):
        source, target = batch
        source, target = source.squeeze(0), target.squeeze(0)
        batch_size = source.size(1)
        model = self.model

        prev_hidden = self.prev_hidden
        if prev_hidden is None:
            prev_hidden = model.init_hidden(batch_size)

        output, hidden = model(source, prev_hidden)
        hidden.detach_()
        self.prev_hidden = hidden
        loss = loss_fn(output, target.view(-1))
        return loss

    def _reduce_eval_outputs(self, outputs):
        total_loss = sum(x['loss'] for x in outputs)
        total_samples = sum(x['num_samples'] for x in outputs)
        avg_loss = total_loss / total_samples
        perplexity = np.exp(avg_loss)
        bpc = np.log2(perplexity)
        return avg_loss, perplexity, bpc

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        lr_scheduler = LambdaLR(optimizer, get_noam_lr_scheduler(self.args))
        self.optimizer = optimizer
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        loss = self._compute_batch(batch, self.train_loss_fn)
        if self.global_step % self.args.log_period == 0:
            self.logger.log_metrics({'loss/train': loss.item()}, self.global_step)
        return loss

    def training_epoch_end(self, outputs):
        # reset prev_hidden to None for the next epoch
        self.prev_hidden = None
        avg_loss = sum(x['loss'].item() for x in outputs) / len(outputs)
        self.logger.log_metrics({'epoch_loss/train': avg_loss}, self.global_step)

    def validation_step(self, batch, batch_idx):
        loss = self._compute_batch(batch, self.eval_loss_fn)
        return {'loss': loss.sum().item(),
                'num_samples': loss.numel()}

    def validation_epoch_end(self, outputs):
        self.prev_hidden = None
        avg_loss, ppl, bpc = self._reduce_eval_outputs(outputs)
        metrics = {'perplexity/valid': ppl,
                   'bpc/valid': bpc,
                   'epoch_loss/valid': avg_loss}
        self.logger.log_metrics(metrics, self.global_step)
        if bpc < self.best_bpc:
            self._save_checkpoint()

    def test_step(self, batch, batch_idx):
        loss = self._compute_batch(batch, self.eval_loss_fn)
        return {'loss': loss.sum().item(),
                'num_samples': loss.numel()}

    def test_epoch_end(self, outputs):
        self.prev_hidden = None
        avg_loss, ppl, bpc = self._reduce_eval_outputs(outputs)
#        self.log('perplexity/test', ppl)
#        self.log('bpc/test', bpc)
#        self.log('epoch_loss/test', avg_loss)
        metrics = {'perplexity/test': ppl,
                   'bpc/test': bpc,
                   'epoch_loss/test': avg_loss}
        self.logger.log_metrics(metrics, self.global_step)

    def on_after_backward(self):
        if self.global_step % self.args.log_period == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                "lr": lr,
            }
            self.logger.log_metrics(metrics, self.global_step)


class Enwik8Runner:
    def __init__(self, args):
        self.args = args

        self.data_module = self.setup_data_module()
        self.args.vocab_size = len(self.data_module._unique)

        self.lit_module = self.setup_model_module()
        self.logger = self.setup_logger()

    def setup_data_module(self):
        data_module = Enwik8DataModule(
            path=self.args.data,
            unroll_size=self.args.unroll_size,
            batch_size=self.args.batch_size,
        )
        data_module.setup()
        return data_module

    def setup_model_module(self):
        lit_module = Enwik8LightningModule(self.args)
        print(lit_module)
        return lit_module

    def setup_logger(self):
        args = self.args
        tb_logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name="{}_{}".format(args.run_name, random.randint(0, 100))
        )
        wuwei_logger = wuwei.Logger(
            project_name=args.project,
            run_name=args.run_name
        )
        loggers = [tb_logger, wuwei_logger]
        return loggers

    def run(self):
        log_dir = self.logger[0].log_dir
        trainer = Trainer(
            logger=self.logger,
            default_root_dir=log_dir,
            max_epochs=self.args.max_epoch,
            gradient_clip_val=self.args.clip_grad,
            log_every_n_steps=self.args.log_period,
            flush_logs_every_n_steps=self.args.log_period,
            gpus=1,
            progress_bar_refresh_rate=0,  # disable progress bar
        )
        self.trainer = trainer
        trainer.fit(model=self.lit_module, datamodule=self.data_module)
        trainer.test(datamodule=self.data_module, verbose=True)
        torch.save(self.lit_module.best_checkpoint,
                   "{}/best_checkpoint.pt".format(log_dir))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--project", type=str, default="FLOP Enwik8 (DEBUG)")
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--run_name", type=str, required=True)
    argparser.add_argument("--noam", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=16000)
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
    argparser.add_argument("--log_period", type=int, default=200)

    args = argparser.parse_args()
    print(args)

    runner = Enwik8Runner(args)
    runner.run()
