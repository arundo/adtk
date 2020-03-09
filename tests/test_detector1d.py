"""Test 1D detectors on some simple cases.

We only test detectors with return_list=False, because return_list=True is
effectively tested by test_list_label_convert.py
"""
from math import isnan

import numpy as np
import pandas as pd
import pytest
from packaging.version import parse
from sklearn.svm import SVR

import adtk.detector as detector
from adtk._base import _TrainableModel
from adtk._utils import PandasBugError

nan = float("nan")

testCases = [
    {
        "model": detector.ThresholdAD,
        "params": {},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.ThresholdAD,
        "params": {"low": -5},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.ThresholdAD,
        "params": {"low": -5, "high": 5},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {"low": 0.1},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {"low": 0.1, "high": 0.9},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 1, 0, 0, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {"c": (None, 3)},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 0, 0, 0, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {"c": (3, None)},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 1, 0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.GeneralizedESDTestAD,
        "params": {},
        "s": [0, -10, 0, 0, 0, nan, 0, 10] + [0] * 10,
        "a": [0, 1, 0, 0, 0, nan, 0, 1] + [0] * 10,
        "pandas_bug": False,
    },
    {
        "model": detector.GeneralizedESDTestAD,
        "params": {"alpha": 0.0001},
        "s": [0, -10, 0, 0, 0, nan, 0, 10] + [0] * 10,
        "a": [0, 0, 0, 0, 0, nan, 0, 0] + [0] * 10,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"side": "positive"},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"side": "negative"},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": 2},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, nan, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, nan, nan] + [0, 0] * 8,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": "50H", "min_periods": 2},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, nan, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, nan, nan] + [0, 0] * 8,
        "pandas_bug": True,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": 2, "min_periods": 1},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, 0, 1] + [0, 0] * 8,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": "50H", "min_periods": 1},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, 0, 1] + [0, 0] * 8,
        "pandas_bug": True,
    },
    {
        "model": detector.PersistAD,
        "params": {"c": 1.5},
        "s": [0, 1, 2.1, 3.3, 4.5, 5.8, 7.3],
        "a": [nan, 0, 0, 0, 0, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": False,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": "72H", "min_periods": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": True,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": ("80H", "72H"), "min_periods": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": True,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": (3, "72H"), "min_periods": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": False,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": ("80H", 3), "min_periods": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": True,
    },
    {
        "model": detector.VolatilityShiftAD,
        "params": {"window": 3},
        "s": [0, 1] * 10 + [100, -100] * 10,
        "a": [nan] * 3 + [0] * 15 + [1] * 5 + [0] * 15 + [nan] * 2,
        "pandas_bug": False,
    },
    {
        "model": detector.VolatilityShiftAD,
        "params": {"window": 3, "agg": "iqr"},
        "s": [0, 1] * 10 + [100, -100] * 10,
        "a": [nan] * 3 + [0] * 15 + [1] * 4 + [0] * 16 + [nan] * 2,
        "pandas_bug": False,
    },
    {
        "model": detector.AutoregressionAD,
        "params": {"n_steps": 2},
        "s": [13, -8, 5, -3, 2, -1, 1, 0, 1, 1, 1, 2, 3, 5, nan, 13, 21, 34],
        "a": [nan, nan, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, nan, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.AutoregressionAD,
        "params": {"n_steps": 2, "side": "positive"},
        "s": [13, -8, 5, -3, 2, -1, 1, 0, 1, 1, 1, 2, 3, 6, 9, 15, 24],
        "a": [nan, nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.AutoregressionAD,
        "params": {"n_steps": 2, "step_size": 2, "side": "negative"},
        "s": [0, 13, 1, -8, 1, 5, 2, -3, 3, 2, 5.1, -1, 8, 1, 13, 0],
        "a": [nan, nan, nan, nan, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.AutoregressionAD,
        "params": {"n_steps": 2, "regressor": SVR(kernel="linear")},
        "s": [13, -8, 5, -3, 2, -1, 1, 0, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        "a": [nan, nan, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.SeasonalAD,
        "params": {},
        "s": [1, 2, 3, 4] * 10 + [1, 3, 2, 4] + [1, 2, 3, 4] * 10,
        "a": [0] * 40 + [0, 1, 1, 0] + [0] * 40,
        "pandas_bug": False,
    },
    {
        "model": detector.SeasonalAD,
        "params": {"freq": 8},
        "s": [1, 2, 3, 4] * 10 + [1, 3, 2, 4] + [1, 2, 3, 4] * 10,
        "a": [0] * 40 + [0, 1, 1, 0] + [0] * 40,
        "pandas_bug": False,
    },
    {
        "model": detector.SeasonalAD,
        "params": {"freq": 8, "trend": True},
        "s": np.array([1, 2, 3, 4] * 10 + [1, 3, 2, 4] + [1, 2, 3, 4] * 10)
        + np.arange(84),
        "a": [nan] * 4 + [0] * 36 + [0, 1, 1, 0] + [0] * 36 + [nan] * 4,
        "pandas_bug": False,
    },
    {
        "model": detector.SeasonalAD,
        "params": {"freq": 8, "trend": True, "side": "positive"},
        "s": np.array([1, 2, 3, 4] * 10 + [1, 3, 2, 4] + [1, 2, 3, 4] * 10)
        + np.arange(84),
        "a": [nan] * 4 + [0] * 36 + [0, 1, 0, 0] + [0] * 36 + [nan] * 4,
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {"detect_func": lambda x: x > 0},
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {
            "detect_func": lambda x, a: x > a,
            "detect_func_params": {"a": 0},
        },
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {
            "detect_func": lambda x, a: x > a,
            "fit_func": lambda x: {"a": x.median()},
        },
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {
            "detect_func": lambda x, a: x > a,
            "fit_func": lambda x, q: {"a": x.quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {
            "detect_func": lambda x, a, b: (x > a) | (x < b),
            "detect_func_params": {"b": -0.5},
            "fit_func": lambda x: {"a": x.median()},
        },
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.CustomizedDetector1D,
        "params": {
            "detect_func": lambda x, a, b: (x > a) | (x < b),
            "detect_func_params": {"b": -0.5},
            "fit_func": lambda x, q: {"a": x.quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "s": [0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        "pandas_bug": False,
    },
]


@pytest.mark.parametrize("testCase", testCases)
def test_fit_detect(testCase):
    """Test fit_detect the detector."""
    s = pd.Series(
        testCase["s"],
        pd.date_range(start="2017-1-1", periods=len(testCase["s"]), freq="D"),
    )
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=s.index)
    if testCase["pandas_bug"] and (parse(pd.__version__) < parse("0.25")):
        with pytest.raises(PandasBugError):
            a = model.fit_detect(s)
    else:
        if isinstance(model, _TrainableModel):
            a = model.fit_detect(s)
        else:
            a = model.detect(s)
        pd.testing.assert_series_equal(a, a_true, check_dtype=False)
        if a_true.sum() == 0:
            assert isnan(model.score(s, a_true, scoring="recall"))
        else:
            assert model.score(s, a_true, scoring="precision") == 1


@pytest.mark.parametrize("testCase", testCases)
def test_fit_and_detect(testCase):
    """Test fit the detector and then detect."""
    s = pd.Series(
        testCase["s"],
        pd.date_range(start="2017-1-1", periods=len(testCase["s"]), freq="D"),
    )
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=s.index)
    if testCase["pandas_bug"] and (parse(pd.__version__) < parse("0.25")):
        with pytest.raises(PandasBugError):
            if isinstance(model, _TrainableModel):
                model.fit(s)
            a = model.detect(s)
    else:
        if isinstance(model, _TrainableModel):
            model.fit(s)
        a = model.detect(s)
        pd.testing.assert_series_equal(a, a_true, check_dtype=False)
        if a_true.sum() == 0:
            assert isnan(model.score(s, a_true, scoring="f1"))
        else:
            assert model.score(s, a_true, scoring="iou") == 1


@pytest.mark.parametrize("testCase", testCases)
def test_dataframe(testCase):
    """Test apply the detector to dataframe."""
    s = pd.Series(
        testCase["s"],
        pd.date_range(start="2017-1-1", periods=len(testCase["s"]), freq="D"),
    )
    df = pd.concat([s.rename("A"), s.rename("B")], axis=1)
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=s.index)
    a_true = pd.concat([a_true.rename("A"), a_true.rename("B")], axis=1)
    if testCase["pandas_bug"] and (parse(pd.__version__) < parse("0.25")):
        with pytest.raises(PandasBugError):
            if isinstance(model, _TrainableModel):
                a = model.fit_detect(df)
            else:
                a = model.detect(df)
    else:
        if isinstance(model, _TrainableModel):
            a = model.fit_detect(df)
        else:
            a = model.detect(df)
        pd.testing.assert_frame_equal(a, a_true, check_dtype=False)


@pytest.mark.parametrize("testCase", testCases)
def test_fit_series_predict_dataframe(testCase):
    """Test fit the detector with a series and predict with dataframe."""
    s = pd.Series(
        testCase["s"],
        pd.date_range(start="2017-1-1", periods=len(testCase["s"]), freq="D"),
    )
    df = pd.concat([s.rename("A"), s.rename("B")], axis=1)
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=s.index)
    a_true = pd.concat([a_true.rename("A"), a_true.rename("B")], axis=1)
    if testCase["pandas_bug"] and (parse(pd.__version__) < parse("0.25")):
        with pytest.raises(PandasBugError):
            if isinstance(model, _TrainableModel):
                model.fit(s)
            a = model.detect(df)
    else:
        if isinstance(model, _TrainableModel):
            model.fit(s)
        a = model.detect(df)
        pd.testing.assert_frame_equal(a, a_true, check_dtype=False)


def test_autoregressive_ad_dataframe():
    """Make sure deepcopy works
    """
    df = pd.DataFrame(
        np.array(
            [
                [13, -8, 5, -3, 2, -1, 1, 0, 1, 1, 2, 3, 6, 9, 15, 24],
                [24, 15, 9, 6, 3, 2, 1, 1, 0, 1, -1, 2, -3, 5, -8, 13],
            ]
        ).T,
        columns=["A", "B"],
        index=pd.date_range(start="2017-1-1", periods=16, freq="D"),
    )
    a_true = pd.DataFrame(
        np.array(
            [
                [nan, nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [nan, nan, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).T,
        columns=["A", "B"],
        index=pd.date_range(start="2017-1-1", periods=16, freq="D"),
    )
    model = detector.AutoregressionAD(
        n_steps=2, side="both", regressor=SVR(kernel="linear")
    )
    for i in range(2):
        a = model.fit_detect(df.iloc[:, i])
        pd.testing.assert_series_equal(a, a_true.iloc[:, i], check_dtype=False)

    a = model.fit_detect(df)
    pd.testing.assert_frame_equal(a, a_true, check_dtype=False)
