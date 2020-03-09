"""Test read-only attributes"""
import numpy as np
import pandas as pd
import pytest

import adtk.detector as detector

testCases = [
    {
        "model": detector.QuantileAD(),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": -float("inf"), "abs_high_": float("inf")},
    },
    {
        "model": detector.QuantileAD(low=0.1),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 1, "abs_high_": float("inf")},
    },
    {
        "model": detector.QuantileAD(high=0.9),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": -float("inf"), "abs_high_": 9},
    },
    {
        "model": detector.QuantileAD(low=0.1, high=0.9),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 1, "abs_high_": 9},
    },
    {
        "model": detector.InterQuartileRangeAD(),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 2.5 - 15, "abs_high_": 7.5 + 15},
    },
    {
        "model": detector.InterQuartileRangeAD(c=2),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 2.5 - 10, "abs_high_": 7.5 + 10},
    },
    {
        "model": detector.InterQuartileRangeAD(c=(2, 4)),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 2.5 - 10, "abs_high_": 7.5 + 20},
    },
    {
        "model": detector.InterQuartileRangeAD(c=(2, None)),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": 2.5 - 10, "abs_high_": float("inf")},
    },
    {
        "model": detector.InterQuartileRangeAD(c=(None, 4)),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": -float("inf"), "abs_high_": 7.5 + 20},
    },
    {
        "model": detector.InterQuartileRangeAD(c=None),
        "s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "a": {"abs_low_": -float("inf"), "abs_high_": float("inf")},
    },
    {
        "model": detector.SeasonalAD(freq=4),
        "s": [0, 1, 2, 1] * 10,
        "a": {"freq_": 4, "seasonal_": [0, 1, 2, 1]},
    },
    {
        "model": detector.SeasonalAD(freq=8),
        "s": [0, 1, 2, 1] * 10,
        "a": {"freq_": 8, "seasonal_": [0, 1, 2, 1, 0, 1, 2, 1]},
    },
    {
        "model": detector.SeasonalAD(),
        "s": [0, 1, 2, 1] * 10,
        "a": {"freq_": 4, "seasonal_": [0, 1, 2, 1]},
    },
    {
        "model": detector.SeasonalAD(trend=True),
        "s": np.array([0, 1, 2, 1] * 10) + np.arange(40) / 10,
        "a": {"freq_": 4, "seasonal_": [-1, 0, 1, 0]},
    },
    {
        "model": detector.SeasonalAD(trend=True, freq=8),
        "s": np.array([0, 1, 2, 1] * 10) + np.arange(40),
        "a": {"freq_": 8, "seasonal_": [-1, 0, 1, 0, -1, 0, 1, 0]},
    },
]


@pytest.mark.parametrize("testCase", testCases)
def test_attribute(testCase):
    """Test fit_detect the detector."""
    s = pd.Series(
        testCase["s"],
        pd.date_range(start="2017-1-1", periods=len(testCase["s"]), freq="D"),
    )
    model = testCase["model"]
    for key, value in testCase["a"].items():
        with pytest.raises(AttributeError):
            getattr(model, key)
    model.fit(s)
    for key, value in testCase["a"].items():
        if isinstance(value, list):
            pd.testing.assert_series_equal(
                getattr(model, key),
                pd.Series(value, index=s.index[: len(value)]),
                check_dtype=False,
                check_names=False,
            )
        else:
            assert getattr(model, key) == value
