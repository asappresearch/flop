from typing import List
from copy import deepcopy

import torch.nn as nn

from flambe.logging import log_histogram
from flop.hardconcrete import HardConcrete
from flop.linear import ProjectedLinear, HardConcreteProjectedLinear, HardConcreteLinear


def make_projected_linear(module: nn.Module, in_place: bool = True) -> nn.Module:
    """Replace all nn.Linear with ProjectedLinear.

    Parameters
    ----------
    module : nn.Module
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    # First find all nn.Linear modules
    modules = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            modules.append((name, child))
        else:
            make_projected_linear(module, in_place)

    # Replace all modules found
    new_module = module if in_place else deepcopy(module)
    for name, child in modules:
        new_child = ProjectedLinear.from_module(child)
        delattr(new_module, name)
        setattr(new_module, name, new_child)

    return new_module


def make_hard_concrete(module: nn.Module,
                       in_place: bool = True,
                       init_mean: float = 0.5,
                       init_std: float = 0.01) -> nn.Module:
    """Replace all ProjectedLinear with HardConcreteProjectedLinear.

    Parameters
    ----------
    module : nn.Module
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    # First find all ProjectedLinear modules
    modules = []
    for name, child in module.named_children():
        if isinstance(child, ProjectedLinear):
            modules.append((name, child))
        elif isinstance(child, nn.Linear):
            modules.append((name, child))
        else:
            make_hard_concrete(child, in_place)

    # Replace all modules found
    new_module = module if in_place else deepcopy(module)
    for name, child in modules:
        if isinstance(child, ProjectedLinear):
            new_child = HardConcreteProjectedLinear.from_module(child, init_mean, init_std)
        else:  # must be nn.Linear
            new_child = HardConcreteLinear.from_module(child, init_mean, init_std)
        delattr(new_module, name)
        setattr(new_module, name, new_child)

    return new_module


def get_hardconcrete_linear_modules(module: nn.Module) -> List[nn.Module]:
    """Get all HardConcrete*Linear modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[nn.Module]
        A list of the HardConcrete*Linear module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, flop.HardConcreteProjectedLinear):
            modules.append(m)
        elif isinstance(m, flop.HardConcreteLinear):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_linear_modules(m))
    return modules


def get_hardconcrete_proj_linear_modules(module: nn.Module) -> List[HardConcreteProjectedLinear]:
    """Get all HardConcreteProjectedLinear modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcreteProjectedLinear]
        A list of the HardConcreteProjectedLinear module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, HardConcreteProjectedLinear):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_proj_linear_modules(m))
    return modules


def get_hardconcrete_modules(module: nn.Module) -> List[HardConcrete]:
    """Get all HardConcrete modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcrete]
        A list of the HardConcrete module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, HardConcrete):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_modules(m))
    return modules


def get_num_prunable_params(modules) -> int:
    return sum([module.num_prunable_parameters() for module in modules])


def get_num_params(modules, train=True) -> int:
    return sum([module.num_parameters(train) for module in modules])


def log_masks(model, masks, step):
    masks = [mask() for mask in masks]
    mask_names = [i[0] for i in model.named_parameters() if "log_alpha" in i[0]]
    alphas = [i[1] for i in model.named_parameters() if "log_alpha" in i[0]]
    for name, log_alpha, mask in zip(mask_names, alphas, masks):
        log_histogram(name, log_alpha.detach().cpu().numpy(), step, bins='sqrt')
        log_histogram(name + ".mask", mask.detach().cpu().numpy(), step, bins='sqrt')
