"""Test HD transformers."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import adtk.transformer as transformer
from adtk._base import _TrainableModel

nan = float("nan")

testCases = [
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {"transform_func": lambda x: x.sum(axis=1) > 0},
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: x.sum(axis=1) > a,
            "transform_func_params": {"a": 0},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: x.sum(axis=1) > a,
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: x.sum(axis=1) > a,
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a, b: (x.sum(axis=1) > a)
            | (x.sum(axis=1) < b),
            "transform_func_params": {"b": -0.5},
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a, b: (x.sum(axis=1) > a)
            | (x.sum(axis=1) < b),
            "transform_func_params": {"b": -0.5},
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "t": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x: pd.DataFrame(
                {"min": x.min(axis=1) > 0, "max": x.max(axis=1) > 0}
            )
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: pd.DataFrame(
                {"min": x.min(axis=1) > a, "max": x.max(axis=1) > a}
            ),
            "transform_func_params": {"a": 0},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: pd.DataFrame(
                {"min": x.min(axis=1) > a, "max": x.max(axis=1) > a}
            ),
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a: pd.DataFrame(
                {"min": x.min(axis=1) > a, "max": x.max(axis=1) > a}
            ),
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a, b: pd.DataFrame(
                {
                    "min": (x.min(axis=1) > a) | (x.min(axis=1) < b),
                    "max": (x.max(axis=1) > a) | (x.max(axis=1) < b),
                }
            ),
            "transform_func_params": {"b": -0.5},
            "fit_func": lambda x: {"a": x.sum(axis=1).median()},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.CustomizedTransformerHD,
        "params": {
            "transform_func": lambda x, a, b: pd.DataFrame(
                {
                    "min": (x.min(axis=1) > a) | (x.min(axis=1) < b),
                    "max": (x.max(axis=1) > a) | (x.max(axis=1) < b),
                }
            ),
            "transform_func_params": {"b": -0.5},
            "fit_func": lambda x, q: {"a": x.sum(axis=1).quantile(q)},
            "fit_func_params": {"q": 0.5},
        },
        "df": [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        ],
        "t": {
            "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "max": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
    },
    {
        "model": transformer.RegressionResidual,
        "params": {"regressor": LinearRegression(), "target": 1},
        "df": [
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, nan, 1, 0],
            [9] * 11,
        ],
        "t": [0] * 8 + [nan] + [0] * 2,
    },
    {
        "model": transformer.PcaProjection,
        "params": {"k": 1},
        "df": [[0, 1, 2, 3, 4, 4, nan, 5, 6], [0, 1, 2, 3, nan, 4, 5, 5, 6]],
        "t": {
            "pc0": [
                3 * 2 ** 0.5,
                2 * 2 ** 0.5,
                1 * 2 ** 0.5,
                0 * 2 ** 0.5,
                nan,
                -1 * 2 ** 0.5,
                nan,
                -2 * 2 ** 0.5,
                -3 * 2 ** 0.5,
            ]
        },
    },
    {
        "model": transformer.PcaReconstruction,
        "params": {"k": 1},
        "df": [
            [0, 1, 2, 3, 3.9, 4.1, 5, 6, 7, 7, 8, 9],
            [0, 1, 2, 3, 4.1, 3.9, 5, 6, 7, nan, 8, 9],
        ],
        "t": {
            0: [0, 1, 2, 3, 4, 4, 5, 6, 7, nan, 8, 9],
            1: [0, 1, 2, 3, 4, 4, 5, 6, 7, nan, 8, 9],
        },
    },
    {
        "model": transformer.PcaReconstructionError,
        "params": {"k": 1},
        "df": [
            [0, 1, 2, 3, 3.9, 4.1, 5, 6, 7, 7, 8, 9],
            [0, 1, 2, 3, 4.1, 3.9, 5, 6, 7, nan, 8, 9],
        ],
        "t": [0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, nan, 0, 0],
    },
]


@pytest.mark.parametrize("testCase", testCases)
def test_fit_transform(testCase):
    """Test fit_transform the transformer."""
    df = pd.DataFrame(
        np.array(testCase["df"]).T,
        pd.date_range(
            start="2017-1-1", periods=len(testCase["df"][0]), freq="D"
        ),
    )
    model = testCase["model"](**testCase["params"])
    if isinstance(model, _TrainableModel):
        t = model.fit_transform(df)
    else:
        t = model.transform(df)
    if not isinstance(testCase["t"], dict):
        t_true = pd.Series(testCase["t"], index=df.index)
        pd.testing.assert_series_equal(t, t_true, check_dtype=False)
    else:
        t_true = pd.DataFrame(testCase["t"], index=df.index)
        pd.testing.assert_frame_equal(t, t_true, check_dtype=False)


@pytest.mark.parametrize("testCase", testCases)
def test_fit_and_transform(testCase):
    """Test fit the transformer and then transform."""
    df = pd.DataFrame(
        np.array(testCase["df"]).T,
        pd.date_range(
            start="2017-1-1", periods=len(testCase["df"][0]), freq="D"
        ),
    )
    model = testCase["model"](**testCase["params"])
    if isinstance(model, _TrainableModel):
        model.fit(df)
    t = model.transform(df)
    if not isinstance(testCase["t"], dict):
        t_true = pd.Series(testCase["t"], index=df.index)
        pd.testing.assert_series_equal(t, t_true, check_dtype=False)
    else:
        t_true = pd.DataFrame(testCase["t"], index=df.index)
        pd.testing.assert_frame_equal(t, t_true, check_dtype=False)
