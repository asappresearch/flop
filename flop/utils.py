from typing import List, Any, Union
from copy import deepcopy

import torch.nn as nn

from flop.hardconcrete import HardConcrete
from flop.linear import (
    ProjectedLinear,
    PrunableModule,
    HardConcreteProjectedLinear,
    HardConcreteLinear,
)


def make_projected_linear(module: nn.Module,
                          in_place: bool = True,
                          keep_weights: bool = False) -> nn.Module:
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
    def make_projected_linear_inplace(module: nn.Module) -> None:
        # First find all nn.Linear modules
        modules = []
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                modules.append((name, child))
            else:
                make_projected_linear_inplace(child)

        # Replace all modules found
        for name, child in modules:
            new_child = ProjectedLinear.from_module(child, keep_weights=keep_weights)
            setattr(module, name, new_child)

    new_module = module if in_place else deepcopy(module)
    make_projected_linear_inplace(new_module)
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
    def make_hard_concrete_inplace(module: nn.Module,
                                   init_mean: float,
                                   init_std: float) -> None:
        modules: List[Any] = []
        for name, child in module.named_children():
            if isinstance(child, ProjectedLinear):
                modules.append((name, child))
            elif isinstance(child, nn.Linear):
                modules.append((name, child))
            else:
                make_hard_concrete_inplace(child, init_mean, init_std)

        for name, child in modules:
            if isinstance(child, ProjectedLinear):
                new_child = HardConcreteProjectedLinear.from_module(child, init_mean, init_std)
            else:  # must be nn.Linear
                new_child = HardConcreteLinear.from_module(child, init_mean, init_std)
            setattr(module, name, new_child)

    new_module = module if in_place else deepcopy(module)
    make_hard_concrete_inplace(new_module, init_mean, init_std)
    return new_module


def make_compressed_module(module: Union[nn.Module, List[nn.Module]],
                           in_place: bool = True) -> Union[nn.Module, List[nn.Module]]:
    """Replace all prunable modules with final compressed module.

    Parameters
    ----------
    module : nn.Module or List[nn.Module]
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    def make_compressed_module_inplace(module: nn.Module) -> None:
        modules: List[Any] = []
        for name, child in module.named_children():
            if isinstance(child, PrunableModule):
                modules.append((name, child))
            else:
                make_compressed_module_inplace(child)

        for name, child in modules:
            new_child = child.to_compressed_module()
            setattr(module, name, new_child)

    new_module = module if in_place else deepcopy(module)
    make_compressed_module_inplace(new_module)
    return new_module


def get_hardconcrete_prunable_modules(module: nn.Module) -> List[PrunableModule]:
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
    modules: List[Any] = []
    for m in module.children():
        if isinstance(m, PrunableModule):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_prunable_modules(m))
    return modules


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
    modules: List[Any] = []
    for m in module.children():
        if isinstance(m, HardConcreteLinear):
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

