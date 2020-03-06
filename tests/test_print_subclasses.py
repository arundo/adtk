import pytest

import adtk.aggregator as aggt
import adtk.detector as detector
import adtk.transformer as transformer


def test_print_subclasses():
    """
    get `print_all_models` method for every module
    """
    _ = aggt.print_all_models()
    _ = detector.print_all_models()
    _ = transformer.print_all_models()
