"""Test HD detectors on some simple cases."""
from math import isnan

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

import adtk.detector as detector
from adtk._base import _TrainableModel

nan = float("nan")

testCases = [
    {
        "model": detector.CustomizedDetectorHD,
        "params": {"detect_func": lambda x: x.sum(axis=1) > 0},
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": detector.CustomizedDetectorHD,
        "params": {
            "detect_func": lambda x, a: x.sum(axis=1) > a,
            "detect_func_params": {"a": 0},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": detector.CustomizedDetectorHD,
        "params": {
            "detect_func": lambda x, a: x.sum(axis=1) > a,
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": detector.CustomizedDetectorHD,
        "params": {
            "detect_func": lambda x, a: x.sum(axis=1) > a,
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": detector.CustomizedDetectorHD,
        "params": {
            "detect_func": lambda x, a, b: (x.sum(axis=1) > a)
            | (x.sum(axis=1) < b),
            "detect_func_params": {"b": -0.5},
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
    {
        "model": detector.CustomizedDetectorHD,
        "params": {
            "detect_func": lambda x, a, b: (x.sum(axis=1) > a)
            | (x.sum(axis=1) < b),
            "detect_func_params": {"b": -0.5},
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
    {
        "model": detector.MinClusterDetector,
        "params": {"model": KMeans(n_clusters=2)},
        "df": [[0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0]],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.MinClusterDetector,
        "params": {"model": KMeans(n_clusters=2)},
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.OutlierDetector,
        "params": {
            "model": LocalOutlierFactor(n_neighbors=1, contamination=0.1)
        },
        "df": [[0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0]],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.OutlierDetector,
        "params": {
            "model": LocalOutlierFactor(n_neighbors=1, contamination=0.1)
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.OutlierDetector,
        "params": {
            "model": IsolationForest(n_estimators=100, contamination=0.1)
        },
        "df": [[0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0]],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.RegressionAD,
        "params": {"target": 2, "regressor": LinearRegression()},
        "df": [
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
            [0, 2, 4, 6, 8, 10, 12, 14, 14, 16, 18],
            [0, 3, 6, 10, 12, 14, 18, 21, nan, 24, 27],
        ],
        "a": [0, 0, 0, 1, 0, 1, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.RegressionAD,
        "params": {
            "target": 2,
            "regressor": LinearRegression(),
            "side": "negative",
        },
        "df": [
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
            [0, 2, 4, 6, 8, 10, 12, 14, 14, 16, 18],
            [0, 3, 6, 10, 12, 14, 18, 21, nan, 24, 27],
        ],
        "a": [0, 0, 0, 0, 0, 1, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.RegressionAD,
        "params": {
            "target": 2,
            "regressor": LinearRegression(),
            "side": "negative",
            "c": 100,
        },
        "df": [
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
            [0, 2, 4, 6, 8, 10, 12, 14, 14, 16, 18],
            [0, 3, 6, 10, 12, 14, 18, 21, nan, 24, 27],
        ],
        "a": [0, 0, 0, 0, 0, 0, 0, 0, nan, 0, 0],
    },
    {
        "model": detector.PcaAD,
        "params": {"k": 1, "c": 3},
        "df": [
            [0, 1, 2, 3, 3.9, 4.1, 5, 6, 7, 7, 8, 9],
            [0, 1, 2, 3, 4.1, 3.9, 5, 6, 7, nan, 8, 9],
        ],
        "a": [0, 0, 0, 0, 1, 1, 0, 0, 0, nan, 0, 0],
    },
]


@pytest.mark.parametrize("testCase", testCases)
def test_fit_detect(testCase):
    """Test fit_detect the detector."""
    df = pd.DataFrame(
        np.array(testCase["df"]).T,
        pd.date_range(
            start="2017-1-1", periods=len(testCase["df"][0]), freq="D"
        ),
    )
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=df.index)
    if isinstance(model, _TrainableModel):
        a = model.fit_detect(df)
    else:
        a = model.detect(df)
    pd.testing.assert_series_equal(a, a_true, check_dtype=False)
    if a_true.sum() == 0:
        assert isnan(model.score(df, a_true, scoring="recall"))
    else:
        assert model.score(df, a_true, scoring="precision") == 1


@pytest.mark.parametrize("testCase", testCases)
def test_fit_and_detect(testCase):
    """Test fit the detector and then detect."""
    df = pd.DataFrame(
        np.array(testCase["df"]).T,
        pd.date_range(
            start="2017-1-1", periods=len(testCase["df"][0]), freq="D"
        ),
    )
    model = testCase["model"](**testCase["params"])
    a_true = pd.Series(testCase["a"], index=df.index)
    if isinstance(model, _TrainableModel):
        model.fit(df)
    a = model.detect(df)
    pd.testing.assert_series_equal(a, a_true, check_dtype=False)
    if a_true.sum() == 0:
        assert isnan(model.score(df, a_true, scoring="f1"))
    else:
        assert model.score(df, a_true, scoring="iou") == 1


@pytest.mark.parametrize("testCase", testCases)
def test_series(testCase):
    """Test the detector on series."""
    if len(testCase["df"]) == 1:
        s = pd.DataFrame(
            testCase["df"][0],
            pd.date_range(
                start="2017-1-1", periods=len(testCase["df"][0]), freq="D"
            ),
        )
        model = testCase["model"](**testCase["params"])
        a_true = pd.Series(testCase["a"], index=s.index)
        if isinstance(model, _TrainableModel):
            a = model.fit_detect(s)
        else:
            a = model.detect(s)
        pd.testing.assert_series_equal(a, a_true, check_dtype=False)
