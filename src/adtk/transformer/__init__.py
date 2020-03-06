"""Module of transformers.

A transformer transforms time series to extract useful information.

"""
from .transformer_1d import *
from .transformer_hd import *
from .transformer_1d import __all__ as __all_1d__
from .transformer_hd import __all__ as __all_hd__

from .._utils import _get_all_subclasses_from_superclass
from .._transformer_base import _Transformer1D, _TransformerHD


def print_all_models():
    """
    Print description of every model in this module.
    """
    model_desc = _get_all_subclasses_from_superclass(_Transformer1D)
    model_desc.update(_get_all_subclasses_from_superclass(_TransformerHD))
    for key, value in model_desc.items():
        print("-" * 80)
        print(key)
        print(value)


__all__ = __all_1d__ + __all_hd__ + ["print_all_models"]
