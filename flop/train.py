from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from flambe.logging import log
from flambe.metric import Metric
from flambe.sampler import Sampler
from flambe.dataset import Dataset
from flambe.learn import Trainer
from flambe.nn import Module
from flambe.optim import NoamScheduler

from flop.utils import make_hard_concrete
from flop.utils import get_hardconcrete_modules, get_hardconcrete_proj_linear_modules
from flop.utils import get_num_prunable_params, get_num_params, log_masks


class HardConcreteTrainer(Trainer):

    def __init__(self,
                 dataset: Dataset,
                 train_sampler: Sampler,
                 val_sampler: Sampler,
                 model: Module,
                 loss_fn: Metric,
                 metric_fn: Metric,
                 optimizer,
                 scheduler=None,
                 device: Optional[str] = None,
                 max_steps: int = 10,
                 epoch_per_step: float = 1.0,
                 iter_per_step: Optional[int] = None,
                 batches_per_iter: int = 1,
                 lower_is_better: bool = False,
                 max_grad_norm: Optional[float] = None,
                 max_grad_abs_val: Optional[float] = None,
                 extra_validation_metrics: Optional[List[Metric]] = None,
                 lr_warmup: int = 100,
                 model_dim: int = 512,
                 iter_before_pruning: int = 0,
                 init_mean: float = 0.5,
                 init_std: float = 0.01,
                 alphas_lr: float = 0.001,
                 lambdas_lr: float = 1.0,
                 target_sparsity: float = 0.8,
                 target_sparsity_warmup: int = 80000,
                 weight_decay: float = 0) -> None:
        """Initialize the Trainer.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the first N columns of data for the
            student model, and the last N columns for the target.
        train_sampler : Sampler
            The sampler to use over training examples
        val_sampler : Sampler
            The sampler to use over validation examples
        model : Module
            The model to train
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler
        device: str, optional
            The device to use in the computation. Only used by compile.
        max_steps : int, optional
            The maximum number of training steps to run
        epoch_per_step : float, optional
            Fraction of an epoch to perform in a single training step
            (i.e before a checkpoint.) Defaults to 1.
            Overriden by `iter_per_step`, if given.
        iter_per_step : int, optional
            Number of iterations to perform in a single training step.
            Overrides `epoch_per_step` if given.
        batches_per_iter : int, optional
            Number of batches to pass through the model before
            calling optimizer.step. Requires the sampler to have
            drop_last set to True. (default set to 1 so optimizer.step
            is called after every batch)
        lower_is_better : bool, optional
            If true, the lowest dev metric is considered best,
            otherwise the highest. Defaults to False.
        max_grad_norm : float, optional
            Maximum Euclidean norm of gradient after clipping.
        max_grad_abs_val: float, optional
            Maximum absolute value of all gradient vector components
            after clipping.
        extra_validation_metrics: Optional[List[Metric]]
            A list with extra metrics to show in each step
            but which don't guide the training procedures
            (i.e model selection through early stopping)

        """
        super().__init__(dataset,
                         train_sampler,  # type: ignore
                         val_sampler,
                         model,
                         loss_fn,
                         metric_fn,
                         optimizer,
                         scheduler,
                         device,
                         max_steps,
                         epoch_per_step,
                         iter_per_step,
                         batches_per_iter,
                         lower_is_better,
                         max_grad_norm,
                         max_grad_abs_val,
                         extra_validation_metrics)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lambda_1 = nn.Parameter(torch.tensor(0.).to(device))  # type: ignore
        self.lambda_2 = nn.Parameter(torch.tensor(0.).to(device))  # type: ignore

        self.optimizer_lambdas = Adam([self.lambda_1, self.lambda_2], weight_decay=0)
        self.optimizer_lambdas.param_groups[0]['lr'] = -lambdas_lr  # type: ignore
        self.lambdas_scheduler = NoamScheduler(self.optimizer_lambdas,
                                               warmup=lr_warmup,
                                               d_model=model_dim)

        self.iter_before_pruning = iter_before_pruning
        self.init_num_params = sum([len(p.view(-1)) for p in self.model.parameters()])

        make_hard_concrete(self.model, in_place=True, init_mean=init_mean, init_std=init_std)

        self.hard_concrete_modules = get_hardconcrete_proj_linear_modules(self.model)
        self.hard_concrete_masks = get_hardconcrete_modules(self.model)
        self.max_prunable = get_num_prunable_params(self.hard_concrete_modules)
        self.target_sparsity = max(min(target_sparsity, 1.0), 0.0)
        self.target_sparsity_warmup = target_sparsity_warmup

        self.model.to(device)

        model_params = (p for n, p in self.model.named_parameters() if 'log_alpha' not in n)
        alpha_params = (p for n, p in self.model.named_parameters() if 'log_alpha' in n)

        self.optimizer_alphas = Adam(alpha_params,
                                     lr=alphas_lr)  # type: ignore
        self.alphas_scheduler = NoamScheduler(self.optimizer_alphas,
                                              warmup=lr_warmup,
                                              d_model=model_dim)

        self.optimizer = Adam(model_params,
                              lr=self.optimizer.param_groups[0]['lr'],  # type: ignore
                              weight_decay=weight_decay)
        self.lr_scheduler = NoamScheduler(self.optimizer,
                                          warmup=lr_warmup,
                                          d_model=model_dim)

        log('Total_params', int(self.init_num_params), 0)
        log('Max_prunable', int(self.max_prunable), 0)

    def _train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()

        tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

        with torch.enable_grad():
            for i in range(self.iter_per_step):
                # Zero the gradients and clear the accumulated loss
                self.optimizer.zero_grad()
                self.optimizer_alphas.zero_grad()
                self.optimizer_lambdas.zero_grad()

                accumulated_loss = 0.0
                for _ in range(self.batches_per_iter):
                    # Get next batch
                    batch = next(self._train_iterator)
                    batch = self._batch_to_device(batch)

                    # Compute loss
                    loss = self._compute_loss(batch) / self.batches_per_iter
                    accumulated_loss += loss.item()
                    loss.backward()

                # Log loss
                global_step = (self.iter_per_step * self._step) + i

                # Clip gradients if necessary
                if self.max_grad_norm:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.max_grad_abs_val:
                    clip_grad_value_(self.model.parameters(), self.max_grad_abs_val)

                log(f'{tb_prefix}Training/Loss', accumulated_loss, global_step)
                log(f'{tb_prefix}Training/Gradient_Norm', self.model.gradient_norm, global_step)
                log(f'{tb_prefix}Training/Parameter_Norm', self.model.parameter_norm, global_step)

                if global_step >= self.iter_before_pruning:

                    pruning_step = global_step - self.iter_before_pruning

                    num_parameters = get_num_params(self.hard_concrete_modules, train=True)
                    expected_sparsity = 1. - (num_parameters / self.max_prunable)

                    if self.target_sparsity_warmup > 0:
                        factor = min(1.0, pruning_step / self.target_sparsity_warmup)
                        target_sparsity = self.target_sparsity * factor
                    else:
                        target_sparsity = self.target_sparsity

                    lagrangian_loss = self.lambda_1 * (target_sparsity - expected_sparsity)
                    lagrangian_loss += self.lambda_2 * (target_sparsity - expected_sparsity) ** 2
                    lagrangian_loss.backward()
                    log("Expected_sparsity", float(expected_sparsity), global_step)
                    log("Lagrangian_loss", lagrangian_loss.item(), global_step)
                    log("Target_sparsity", target_sparsity, global_step)
                    log("lambda_1", self.lambda_1.item(), global_step)
                    log("lambda_2", self.lambda_2.item(), global_step)

                    self.optimizer_lambdas.step()
                    self.lambdas_scheduler.step(pruning_step)

                    self.optimizer_alphas.step()
                    self.alphas_scheduler.step(pruning_step)

                # Optimize
                self.optimizer.step()
                self.lr_scheduler.step(global_step)

            # Zero the gradients when exiting a train step
            self.optimizer.zero_grad()
            self.optimizer_lambdas.zero_grad()
            self.optimizer_alphas.zero_grad()

    def _eval_step(self) -> None:
        super()._eval_step()
        log_masks(self.model, self.hard_concrete_masks, self._step)
        num_parameters = get_num_params(self.hard_concrete_modules, train=False)
        num_non_prunable = self.init_num_params - self.max_prunable
        total_num_params = int(num_parameters) + num_non_prunable
        relative_sparsity = 1. - (num_parameters / self.max_prunable)
        log("Num_Params", int(num_parameters), self._step)
        log("Relative_sparsity", float(relative_sparsity), self._step)
        log("True_sparsity", 1. - total_num_params / self.init_num_params, self._step)
        log("Total_num_params", total_num_params, self._step)
        log('LambdaLR', self.optimizer_lambdas.param_groups[0]['lr'], self._step)  # type: ignore
        log('AlphaLR', self.optimizer_alphas.param_groups[0]['lr'], self._step)  # type: ignore

    def _state(self,
               state_dict,
               prefix: str,
               local_metadata: Dict[str, Any]):
        state_dict[prefix + 'lambda_1'] = self.lambda_1.data  # type: ignore
        state_dict[prefix + 'lambda_2'] = self.lambda_2.data  # type: ignore
        state_dict[prefix + 'optimizer'] = self.optimizer.state_dict()
        state_dict[prefix + 'optimizer_lambdas'] = self.optimizer_lambdas.state_dict()
        state_dict[prefix + 'optimizer_alphas'] = self.optimizer_alphas.state_dict()
        state_dict[prefix + 'lr_scheduler'] = self.lr_scheduler.state_dict()
        state_dict[prefix + 'lambdas_scheduler'] = self.lambdas_scheduler.state_dict()
        state_dict[prefix + 'alphas_scheduler'] = self.alphas_scheduler.state_dict()
        return state_dict

    def _load_state(self,
                    state_dict,
                    prefix: str,
                    local_metadata: Dict[str, Any],
                    strict: bool,
                    missing_keys: List[Any],
                    unexpected_keys: List[Any],
                    error_msgs: List[Any]) -> None:
        self.lambda_1.data = state_dict[prefix + 'lambda_1']  # type: ignore
        self.lambda_2.data = state_dict[prefix + 'lambda_2']  # type: ignore
        self.optimizer.load_state_dict(state_dict[prefix + 'optimizer'])
        self.optimizer_alphas.load_state_dict(state_dict[prefix + 'optimizer_alphas'])
        self.optimizer_lambdas.load_state_dict(state_dict[prefix + 'optimizer_lambdas'])
        self.lr_scheduler.load_state_dict(state_dict[prefix + 'lr_scheduler'])
        self.lambdas_scheduler.load_state_dict(state_dict[prefix + 'lambdas_scheduler'])
        self.alphas_scheduler.load_state_dict(state_dict[prefix + 'alphas_scheduler'])
