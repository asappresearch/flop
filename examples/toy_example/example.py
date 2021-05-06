from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import flop


# given the original model (such as BERT),
# (a) replace all nn.Linear() in it with flop.ProjectedLinear 
# (b) convert ProjectedLinear into HardConcreteProjectLinear that inserts a pruning mask
def initialize_model(original_model: nn.Module) -> nn.Module:
    # keep_weights=True to apply SVD and initialize factorization with SVD outputs
    model = flop.make_projected_linear(original_model, in_place=False, keep_weights=True)
    model = flop.make_hard_concrete(model, in_place=True)
    return model


# return lagrangian variables and the associated optimizer
# IMPORTANT: the learning rate is set to negative so lagrangian variables maximize the penalty.
def initialize_lagrangian(lambda_lr: float) -> Tuple:
    lambda_1 = nn.Parameter(torch.tensor(0.))
    lambda_2 = nn.Parameter(torch.tensor(0.))
    optimizer_lambda = optim.Adam(
        [lambda_1, lambda_2],
    )
    optimizer_lambda.param_groups[0]['lr'] = -lambda_lr
    return lambda_1, lambda_2, optimizer_lambda


# compute lagrangian loss (lagrangian penalty) that's the 2nd term of training loss
# return the loss as a differentiable scalar and a dictionary of metrics that can be logged
def compute_training_compression_loss_and_metrics(hardconcrete_modules: List[flop.PrunableModule],
                                                  lambda_1: nn.Parameter,
                                                  lambda_2: nn.Parameter,
                                                  global_step: int,
                                                  pruning_warmup_steps: int,
                                                  target_compression_rate: float):
    if pruning_warmup_steps > 0:
        rate = min(1.0, global_step / pruning_warmup_steps)
        target_compression_rate = target_compression_rate * rate

    expected_size = sum(m.num_parameters(train=True) for m in hardconcrete_modules)
    prunable_size = sum(m.num_prunable_parameters() for m in hardconcrete_modules)
    expected_compression_rate = 1.0 - expected_size / prunable_size

    lagrangian_loss = (
        lambda_1 * (expected_compression_rate - target_compression_rate) +
        lambda_2 * (expected_compression_rate - target_compression_rate) ** 2
    )
    metrics_to_log = {
        'compression_loss': lagrangian_loss.item(),
        'size/expected_size': expected_size.item(),
        'size/prunable_size': prunable_size,
        'compression/target_compression_rate': target_compression_rate,
        'compression/expected_compression_rate': expected_compression_rate.item(),
        'lambda_1': lambda_1.item(),
        'lambda_2': lambda_2.item(),
    }
    return lagrangian_loss, metrics_to_log

# compute inference time compression metrics
# call it after evaluation
def compute_validation_compression_metrics(hardconcrete_modules: List[flop.PrunableModule]):
    inference_size = sum(m.num_parameters(train=False) for m in hardconcrete_modules)
    prunable_size = sum(m.num_prunable_parameters() for m in hardconcrete_modules)
    inference_compression_rate = 1.0 - inference_size / prunable_size
    return {
        'size/inference_size': inference_size.item(),
        'compression/inference_compression_rate': inference_compression_rate.item(),
    }


# hyperparameters of the pruning training
model_lr = 0.001
lambda_lr = 0.0003
pruning_warmup_steps = 16000
target_compression_rate = 0.5  # prune 30% of the prunable weights

# a toy example of original model
original_model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
)

# step 1: convert original model into a prunable model w/ HardConcrete modules
model = initialize_model(original_model)
hardconcrete_modules = flop.get_hardconcrete_prunable_modules(model)

# step 2: initialize Lagrangian variables and Adam optimizers
lambda_1, lambda_2, optimizer_lambda = initialize_lagrangian(lambda_lr)
optimizer = optim.Adam(
    list(model.parameters())
)

# step 3: start training
for global_step in range(1, 40000 + 1):
    # a random input
    x = torch.rand(5, 10)

    # compute task specific loss given input x
    logits = model(x)
    task_loss = ((logits - 1.0) ** 2).sum()

    # compute lagrangian loss / penalty
    lagrangian_loss, metrics_to_log = compute_training_compression_loss_and_metrics(
        hardconcrete_modules,
        lambda_1,
        lambda_2,
        global_step,
        pruning_warmup_steps,
        target_compression_rate
    )

    # total loss and backward 
    loss = task_loss + lagrangian_loss
    loss.backward()

    # optimization step
    optimizer.step()
    optimizer_lambda.step()
    optimizer.zero_grad()
    optimizer_lambda.zero_grad()

    # log loss and metrics_to_log
    if global_step % 1000 == 0:
        # log sth to tensorboard
        print(loss, task_loss, lagrangian_loss)
        pass

    # evaluation step
    if global_step % 1000 == 0:
        model.eval()
        # do evaluation here and log sth to tensorboard
        # eval_model() ..
        # eval_compression_metrics = compute_validation_compression_metrics(hardconcrete_modules)
        model.train()

# step 4: convert the prunable model into a final dense model (w/o hardconcrete modules)
compressed_model = flop.make_compressed_module(model, in_place=False)
