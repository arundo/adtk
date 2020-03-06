"""Module of detectors.

A detector detects anomalous time points from time series.

"""
from .detector_1d import *
from .detector_hd import *
from .._detector_base import _Detector1D
from .._detector_base import _DetectorHD
from .detector_1d import __all__ as __all_1d__
from .detector_hd import __all__ as __all_hd__

from .._utils import _get_all_subclasses_from_superclass


def print_all_models():
    """
    Print description of every model in this module.
    """
    model_desc = _get_all_subclasses_from_superclass(_Detector1D)
    model_desc.update(_get_all_subclasses_from_superclass(_DetectorHD))
    for key, value in model_desc.items():
        print("-" * 80)
        print(key)
        print(value)


__all__ = __all_1d__ + __all_hd__ + ["print_all_models"]
