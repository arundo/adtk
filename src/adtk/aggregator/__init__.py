"""Module of aggregators.

An aggregator combines multiple lists of anomalies into one.

"""
from ._aggregator import OrAggregator, AndAggregator, CustomizedAggregator


from .._utils import _get_all_subclasses_from_superclass
from .._aggregator_base import _Aggregator

from typing import Dict, Optional


def print_all_models() -> None:
    """
    Print description of every model in this module.
    """
    model_desc = _get_all_subclasses_from_superclass(
        _Aggregator
    )  # type: Dict[str, Optional[str]]
    for key, value in model_desc.items():
        print("-" * 80)
        print(key)
        print(value)


__all__ = [
    "OrAggregator",
    "AndAggregator",
    "CustomizedAggregator",
    "print_all_models",
]
