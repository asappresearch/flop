
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Dict
import warnings

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import LightningModule

import flop


class CompressionModule(LightningModule, ABC):

    # Avoid mypy error, e.g.,
    # https://github.com/python/mypy/issues/8795
    def _setup_model_for_compression(self, *input: Any) -> None:
        raise NotImplementedError

    setup_model_for_compression: Callable[..., Any] = _setup_model_for_compression

    @abstractmethod
    def compute_training_compression_loss_and_metrics(
        self,
        batch: Optional[Tensor],
        batch_idx: Optional[int],
     ) -> Tuple[Tensor, Dict]:
        pass

    @abstractmethod
    def compute_validation_compression_metrics(
        self,
        batch: Optional[Tensor],
        batch_idx: Optional[int],
    ) -> Dict:
        pass

    @abstractmethod
    def extra_optimization_step(self) -> Any:
        pass

    @abstractmethod
    def finalize_compression(self) -> Any:
        pass


class FlopPruningModule(CompressionModule):

    # ---------------------------------------------------------------
    # Implementations of abstract methods
    # ---------------------------------------------------------------

    def setup_model_for_compression(
        self,
        pruning_warmup_steps: int = 64000,
        pruning_target: float = 0.7,
        lambda_lr: float = 0.0001,
        hardconcrete_mean: float = 0.5,
        hardconcrete_std: float = 0.1,
        squared_penalty_coeff: float = 2.0,
    ):
        self.pruning_warmup_steps = pruning_warmup_steps
        self.pruning_target = pruning_target
        self.lambda_lr = lambda_lr
        self.squared_penalty_coeff = squared_penalty_coeff

        flop.make_hard_concrete(
            self.model,
            in_place=True,
            init_mean=hardconcrete_mean,
            init_std=hardconcrete_std,
        )
        self.hardconcrete_modules = flop.get_hardconcrete_prunable_modules(self.model)
        self.lambda_1 = nn.Parameter(torch.tensor(0.))
        self.lambda_2 = nn.Parameter(torch.tensor(0.))
        self.optimizer_lambda = self.configure_compression_optimizer()

    def configure_compression_optimizer(self):
        optimizer_lambda = Adam(
            [self.lambda_1, self.lambda_2],
            lr=self.lambda_lr
        )
        optimizer_lambda.param_groups[0]['lr'] = -self.lambda_lr
        return optimizer_lambda

    def compute_training_compression_loss_and_metrics(self, batch, batch_idx):
        if self.global_step > 0:
            self.optimizer_lambda.step()
            self.optimizer_lambda.zero_grad()

        target_compression_rate = self.pruning_target
        if self.pruning_warmup_steps > 0:
            rate = min(1.0, self.global_step / self.pruning_warmup_steps)
            target_compression_rate = self.pruning_target * rate

        hardconcrete_modules = self.hardconcrete_modules
        expected_size = sum(m.num_parameters(train=True) for m in hardconcrete_modules)
        prunable_size = sum(m.num_prunable_parameters() for m in hardconcrete_modules)
        expected_compression_rate = 1.0 - expected_size / prunable_size

        beta = self.squared_penalty_coeff
        lagrangian_loss = (
            self.lambda_1 * (expected_compression_rate - target_compression_rate) +
            self.lambda_2 * beta * (expected_compression_rate - target_compression_rate) ** 2
        )
        metrics_to_log = {
            'compression_loss': lagrangian_loss.item(),
            'size/expected_size': expected_size.item(),
            'size/prunable_size': prunable_size,
            'compression/target_compression_rate': target_compression_rate,
            'compression/expected_compression_rate': expected_compression_rate.item(),
            'lambda_1': self.lambda_1.item(),
            'lambda_2': self.lambda_2.item(),
        }
        return lagrangian_loss, metrics_to_log

    def compute_validation_compression_metrics(self, batch, batch_idx):
        hardconcrete_modules = self.hardconcrete_modules
        inference_size = sum(m.num_parameters(train=False) for m in hardconcrete_modules)
        prunable_size = sum(m.num_prunable_parameters() for m in hardconcrete_modules)
        inference_compression_rate = 1.0 - inference_size / prunable_size
        return {
            'size/inference_size': inference_size.item(),
            'compression/inference_compression_rate': inference_compression_rate.item(),
        }

    def extra_optimization_step(self):
        pass

    def finalize_compression(self):
        flop.make_compressed_module(
            self.model,
            in_place=True,
        )

    # ---------------------------------------------------------------
    # Overrided PTL methods for automatic compression training
    # ---------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        # call user defined training_step()
        warnings.warn("{}".format(super(CompressionModule, self).training_step))
        result = super(CompressionModule, self).training_step(batch, batch_idx)

        # compute pruning loss and metrics
        compression_loss, compression_metrics = self.compute_training_compression_loss_and_metrics(
            batch,
            batch_idx
        )

        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.logger.log_metrics(compression_metrics, step=self.global_step)

        if isinstance(result, Tensor):
            # returned result is the loss
            result = result + compression_loss
        elif isinstance(result, Dict):
            # returned result is a dictionary
            result['loss'] = result['loss'] + compression_loss
        else:
            raise ValueError("Unrecognized return type from training_step()")

        return result

    def validation_epoch_end(self, outputs):
        # call user defined validation_epoch_end()
        if hasattr(super(CompressionModule, self), 'validation_epoch_end'):
            super(CompressionModule, self).validation_epoch_end(outputs)
        metrics = self.compute_validation_compression_metrics(None, None)
        self.logger.log_metrics(metrics, self.global_step)
