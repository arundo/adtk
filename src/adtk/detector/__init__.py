"""Module of detectors.

A detector detects anomalous time points from time series.

"""
from ._detector_1d import (
    ThresholdAD,
    QuantileAD,
    InterQuartileRangeAD,
    GeneralizedESDTestAD,
    PersistAD,
    LevelShiftAD,
    VolatilityShiftAD,
    AutoregressionAD,
    SeasonalAD,
    CustomizedDetector1D,
)
from ._detector_hd import (
    MinClusterDetector,
    OutlierDetector,
    RegressionAD,
    PcaAD,
    CustomizedDetectorHD,
)
from .._detector_base import (
    _NonTrainableUnivariateDetector,
    # _NonTrainableMultivariateDetector,
    _TrainableUnivariateDetector,
    _TrainableMultivariateDetector,
)


from .._utils import _get_all_subclasses_from_superclass


def print_all_models() -> None:
    """
    Print description of every model in this module.
    """
    model_desc = _get_all_subclasses_from_superclass(
        _NonTrainableUnivariateDetector
    )
    # model_desc.update(
    # _get_all_subclasses_from_superclass(_NonTrainableMultivariateDetector)
    # )
    model_desc.update(
        _get_all_subclasses_from_superclass(_TrainableUnivariateDetector)
    )
    model_desc.update(
        _get_all_subclasses_from_superclass(_TrainableMultivariateDetector)
    )
    for key, value in model_desc.items():
        print("-" * 80)
        print(key)
        print(value)


__all__ = [
    "ThresholdAD",
    "QuantileAD",
    "InterQuartileRangeAD",
    "GeneralizedESDTestAD",
    "PersistAD",
    "LevelShiftAD",
    "VolatilityShiftAD",
    "AutoregressionAD",
    "SeasonalAD",
    "CustomizedDetector1D",
    "MinClusterDetector",
    "OutlierDetector",
    "RegressionAD",
    "PcaAD",
    "CustomizedDetectorHD",
    "print_all_models",
]
