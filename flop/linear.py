from typing import Optional, Tuple
from copy import deepcopy

import math

import torch
import torch.nn as nn
from torch.nn import Module

from .hardconcrete import HardConcrete


class ProjectedLinear(Module):
    """Linear layer with an internal projection."""

    __constants__ = ['in_features', 'out_features', 'proj_features', 'activation']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 proj_features: Optional[int] = None,
                 activation: Optional[nn.Module] = None) -> None:
        """Initialize a HardConcreteProjectedLinear module.

        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool, optional
            Whether to add a bias term, by default True
        proj_features: int, optional
            The numer of dimensions in the intermediate projection.
            By default: ``in_feat * out_feat // (in_feat + out_feat)``,
            which results in the same parameter count as a traditional
            (in_feat x out_feat) weight matrix.
        activation: nn.Module, optional
            An optional activation after the first projection.
            Default ``None``.

        """
        super().__init__()

        if proj_features is None:
            # Compute so that we keep the same number of parameters
            proj_features = in_features * out_features // (in_features + out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features

        self.activation = deepcopy(activation) if activation is not None else None
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, out_features, bias=bias)

    @classmethod
    def from_module(cls,
                    module: nn.Linear,
                    proj_features: Optional[int] = None,
                    activation: Optional[nn.Module] = None,
                    keep_weights: bool = False) -> 'ProjectedLinear':
        """Construct from a nn.Linear module.

        Parameters
        ----------
        module: ProjectedLinear
            A ``Linear`` module.

        Returns
        -------
        ProjectedLinear
            A ProjectedLinear module of the same input and output dim.

        """
        in_features = module.in_features
        out_features = module.out_features
        proj_features = min(in_features, out_features) if keep_weights else None
        bias = module.bias is not None
        new_module = cls(in_features,
                         out_features,
                         bias,
                         proj_features,
                         activation)

        if keep_weights:
            U, S, V = torch.svd(module.weight.data)
            # (proj_features, in_features)
            new_module.linear1.weight.data = V.t() * (S.view(-1, 1) ** 0.5)
            # (out_features, proj_features)
            new_module.linear2.weight.data = U * (S.view(1, -1) ** 0.5)
            if bias:
                new_module.linear2.bias.data = module.bias.data.clone()

        return new_module

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the layer.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape [..., in_features]

        Returns
        -------
        torch.Tensor
            [Tensor of shape [..., out_features]

        """
        if self.activation is not None:
            return self.linear2(self.activation(self.linear1(data)))
        else:
            return self.linear2(self.linear1(data))

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", proj_features={proj_features}"
        s += ", bias={}".format(str(self.linear2.bias is not None))
        if self.activation is not None:
            s += ", activation=" + str(self.activation)
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())


class ColumnSparseLinear(Module):
    """Linear layer that multiplies with a sparse set of input features."""

    __constants__ = ['in_features', 'out_features', 'used_in_features']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 indices: torch.Tensor,
                 bias: bool = True) -> None:
        """Initialize a ColumnSparseLinear module.

        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        indices: torch.Tensor
            The column indices (or equivalently input feature indices) used.
        bias : bool, optional
            Whether to add a bias term, by default True

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_used = indices.numel()
        device = indices.device
        self.weight = nn.Parameter(
            torch.zeros(out_features, self.in_features_used, device=device)
        )
        self.register_buffer('indices', indices.data)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the layer.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape [..., in_features]

        Returns
        -------
        torch.Tensor
            [Tensor of shape [..., out_features]

        """
        indices: Optional[torch.Tensor] = self.indices
        if indices is not None:
            data = data.index_select(-1, indices)
        output = data.matmul(self.weight.t())
        return output if self.bias is None else output + self.bias

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())


class PrunableModule(Module):
    """The base class of pytorch prunable modules."""

    def to_compressed_module(self):
        raise NotImplementedError()


class HardConcreteProjectedLinear(PrunableModule):
    """The hard concrete equivalent of ``ProjectedLinear``."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 init_mean: float = 0.5,
                 init_std: float = 0.01,
                 proj_features: Optional[int] = None) -> None:
        """Initialize a HardConcreteProjectedLinear module.

        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool, optional
            Whether to add a bias term, by default True
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        proj_features: int
            The number of dimensions in the hidden projection. Defaults
            to conserving the same number of parameters as a Linear
            layer, which implies a projected dimension of size:
            `in_features * out_features / (in_features + out_features)`

        """
        super().__init__()

        if proj_features is None:
            # Compute so that we keep the same number of parameters
            proj_features = in_features * out_features // (in_features + out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features

        self.weight = nn.Parameter(torch.zeros(proj_features, in_features))  # type: ignore
        self.weight_proj = nn.Parameter(torch.zeros(out_features, proj_features))  # type: ignore
        self.mask = HardConcrete(proj_features, init_mean, init_std)  # type: ignore

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter('bias', None)  # type: ignore

        self.indices: Optional[torch.Tensor] = None
        self.compiled_weight: Optional[torch.Tensor] = None
        self.compiled_weight_proj: Optional[torch.Tensor] = None
        self.reset_parameters()

    @classmethod
    def from_module(cls,
                    module: ProjectedLinear,
                    init_mean: float = 0.5,
                    init_std: float = 0.01,
                    keep_weights: bool = True) -> 'HardConcreteProjectedLinear':
        """Construct from a pretrained ProjectedLinear module.

        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.

        Parameters
        ----------
        module: ProjectedLinear
            A ``ProjectedLinear`` module.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.

        Returns
        -------
        HardConreteProjectedLinear
            The input module with a hardconcrete mask introduced.

        """
        in_features = module.in_features
        out_features = module.out_features
        bias = module.linear2.bias is not None
        proj_features = module.proj_features
        new_module = cls(in_features,
                         out_features,
                         bias,
                         init_mean,
                         init_std,
                         proj_features)

        if keep_weights:
            new_module.weight.data = module.linear1.weight.data.clone()
            new_module.weight_proj.data = module.linear2.weight.data.clone()
            if bias:
                new_module.bias.data = module.linear2.bias.data.clone()

        return new_module

    def reset_parameters(self):
        """Reset network parameters."""
        self.mask.reset_parameters()
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_proj)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return self.in_features * self.proj_features + self.out_features * self.proj_features

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        params = self.weight.new_tensor(0)
        if train:
            n_proj = self.mask.l0_norm()
            params += self.in_features * n_proj + self.out_features * n_proj
        elif self.compiled_weight is not None and self.compiled_weight_proj is not None:
            params += self.compiled_weight.numel() + self.compiled_weight_proj.numel()
        else:
            raise Exception("Weights are not compiled when calling num_parameters().")
        return params

    def _compute_compiled_weights(self,
                                  mask: torch.Tensor,
                                  indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compile and return pruned dense weight matrices."""
        compiled_weight = self.weight.index_select(0, indices)
        compiled_weight_proj = (self.weight_proj * mask).index_select(1, indices)
        return compiled_weight, compiled_weight_proj

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.

        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`

        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`

        """
        if self.training:
            # First reset the compiled weights
            self.compiled_weight = None
            self.compiled_weight_proj = None
            self.indices = None

            # Sample, and compile dynamically
            mask = self.mask()
            indices = mask.data.nonzero().view(-1)
            if indices.numel() > self.proj_features * 0.8:
                compiled_weight = self.weight
                compiled_weight_proj = self.weight_proj * mask
            else:
                compiled_weight, compiled_weight_proj = self._compute_compiled_weights(
                    mask, indices
                )
        else:
            #  Save compiled weights for eval
            if self.compiled_weight is None:
                mask = self.mask()
                indices = mask.nonzero().view(-1)
                self.indices = indices
                self.compiled_weight, self.compiled_weight_proj = self._compute_compiled_weights(
                    mask, indices
                )
            compiled_weight = self.compiled_weight
            compiled_weight_proj = self.compiled_weight_proj

        U = data.matmul(compiled_weight.t()).matmul(compiled_weight_proj.t())
        return U if self.bias is None else U + self.bias

    def to_compressed_module(self) -> ProjectedLinear:
        is_train_mode = self.training
        self.eval()
        mask = self.mask()
        indices = mask.nonzero().view(-1)
        self.train(mode=is_train_mode)
        compiled_weight, compiled_weight_proj = self._compute_compiled_weights(mask, indices)
        module = ProjectedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            proj_features=indices.numel(),
        )
        assert module.linear1.weight.size() == compiled_weight.size()
        assert module.linear2.weight.size() == compiled_weight_proj.size()
        module.linear1.weight.data = compiled_weight.data.clone()
        module.linear2.weight.data = compiled_weight_proj.data.clone()
        if self.bias is not None:
            module.linear2.bias.data = self.bias.data.clone()
        return module

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", proj_features={proj_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())


class HardConcreteLinear(PrunableModule):
    """The hard concrete equivalent of ``nn.Linear``."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 init_mean: float = 0.5,
                 init_std: float = 0.01) -> None:
        """Initialize a HardConcreteLinear module.

        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool, optional
            Whether to add a bias term, by default True
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(in_features, out_features))  # type: ignore
        self.mask = HardConcrete(in_features, init_mean, init_std)  # type: ignore

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter('bias', None)  # type: ignore

        self.indices: Optional[torch.Tensor] = None
        self.compiled_weight: Optional[torch.Tensor] = None
        self.reset_parameters()

    @classmethod
    def from_module(cls,
                    module: nn.Linear,
                    init_mean: float = 0.5,
                    init_std: float = 0.01,
                    keep_weights: bool = True) -> 'HardConcreteLinear':
        """Construct from a pretrained nn.Linear module.

        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.

        Parameters
        ----------
        module: nn.Linear
            A ``nn.Linear`` module.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.

        Returns
        -------
        HardConreteLinear
            The input module with a hardconcrete mask introduced.

        """
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = cls(in_features,
                         out_features,
                         bias,
                         init_mean,
                         init_std)

        if keep_weights:
            new_module.weight.data = module.weight.data.transpose(0, 1).clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()

        return new_module

    def reset_parameters(self):
        """Reset network parameters."""
        self.mask.reset_parameters()
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return self.in_features * self.out_features

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        params = self.weight.new_tensor(0)
        if train:
            n_active = self.mask.l0_norm()
            params += n_active * self.out_features
        elif self.compiled_weight is not None:
            params += self.compiled_weight.numel()
        else:
            raise Exception("Weights are not compiled when calling num_parameters().")
        return params

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.

        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`

        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`

        """
        if self.training:
            # First reset the compiled weights
            self.compiled_weight = None
            self.indices = None

            # Sample, and compile dynamically
            mask = self.mask()
            indices = mask.data.nonzero().view(-1)
            if indices.numel() > self.in_features * 0.8:
                compiled_weight = self.weight * mask.view(-1, 1)
                U = data.matmul(compiled_weight)
            else:
                compiled_weight = self.weight * mask.view(-1, 1)
                compiled_weight = compiled_weight.index_select(0, indices)
                U = data.index_select(-1, indices).matmul(compiled_weight)
        else:
            if self.compiled_weight is None:
                mask = self.mask()
                indices = mask.nonzero().view(-1)
                self.indices = indices
                weight = self.weight * mask.view(-1, 1)
                self.compiled_weight = weight.index_select(0, indices)

            # Use the precompued sub weight
            U = data.index_select(-1, self.indices).matmul(self.compiled_weight)  # type: ignore

        return U if self.bias is None else U + self.bias

    def to_compressed_module(self) -> ColumnSparseLinear:
        is_train_mode = self.training
        self.eval()
        mask = self.mask()
        indices = mask.nonzero().view(-1)
        self.train(mode=is_train_mode)
        compiled_weight = (self.weight * mask.view(-1, 1)).index_select(0, indices)
        compiled_weight = compiled_weight.t()
        module = ColumnSparseLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            indices=indices,
            bias=self.bias is not None,
        )
        assert module.weight.size() == compiled_weight.size()
        module.weight.data = compiled_weight.contiguous().data.clone()
        if self.bias is not None:
            module.bias.data = self.bias.data.clone()
        return module

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())

