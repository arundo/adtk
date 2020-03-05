"""Module of transformers.

A transformer transforms time series to extract useful information.

"""
from ._transformer_1d import (
    RollingAggregate,
    DoubleRollingAggregate,
    ClassicSeasonalDecomposition,
    Retrospect,
    StandardScale,
    CustomizedTransformer1D,
)
from ._transformer_hd import (
    RegressionResidual,
    PcaProjection,
    PcaReconstruction,
    PcaReconstructionError,
    SumAll,
    CustomizedTransformerHD,
)


from .._utils import _get_all_subclasses_from_superclass
from .._transformer_base import (
    _NonTrainableUnivariateTransformer,
    _NonTrainableMultivariateTransformer,
    _TrainableUnivariateTransformer,
    _TrainableMultivariateTransformer,
)


def print_all_models() -> None:
    """
    Print description of every model in this module.
    """
    model_desc = _get_all_subclasses_from_superclass(
        _NonTrainableUnivariateTransformer
    )
    model_desc.update(
        _get_all_subclasses_from_superclass(
            _NonTrainableMultivariateTransformer
        )
    )
    model_desc.update(
        _get_all_subclasses_from_superclass(_TrainableUnivariateTransformer)
    )
    model_desc.update(
        _get_all_subclasses_from_superclass(_TrainableMultivariateTransformer)
    )
    for key, value in model_desc.items():
        print("-" * 80)
        print(key)
        print(value)


__all__ = [
    "RollingAggregate",
    "DoubleRollingAggregate",
    "ClassicSeasonalDecomposition",
    "Retrospect",
    "StandardScale",
    "CustomizedTransformer1D",
    "RegressionResidual",
    "PcaProjection",
    "PcaReconstruction",
    "PcaReconstructionError",
    "SumAll",
    "CustomizedTransformerHD",
    "print_all_models",
]
