"""Test pipeline and pipenet.

We do not test graph plot because it is covered by test_visualization."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import adtk.aggregator as aggregator
import adtk.detector as detector
import adtk.transformer as transformer
from adtk.pipe import Pipeline, Pipenet


def test_pipenet_default():
    """
    Test default setting of pipenet
    """
    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    anomaly = my_pipe.fit_detect(df)
    pd.testing.assert_series_equal(
        anomaly,
        pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
        check_dtype=False,
    )

    assert (
        my_pipe.score(
            df,
            pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
            scoring="recall",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
            scoring="precision",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
            scoring="iou",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
            scoring="f1",
        )
        == 1
    )


def test_pipenet_return_list():
    """
    Test pipenet with return_list=True
    """
    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    anomaly = my_pipe.fit_detect(df, return_list=True)
    assert anomaly == [
        (
            pd.Timestamp("2017-01-05 00:00:00"),
            pd.Timestamp("2017-01-05 23:59:59.999999999"),
        ),
        (
            pd.Timestamp("2017-01-07 00:00:00"),
            pd.Timestamp("2017-01-08 23:59:59.999999999"),
        ),
    ]
    assert (
        my_pipe.score(
            df,
            [
                (
                    pd.Timestamp("2017-01-05 00:00:00"),
                    pd.Timestamp("2017-01-05 23:59:59.999999999"),
                ),
                (
                    pd.Timestamp("2017-01-07 00:00:00"),
                    pd.Timestamp("2017-01-08 23:59:59.999999999"),
                ),
            ],
            scoring="precision",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            [
                (
                    pd.Timestamp("2017-01-05 00:00:00"),
                    pd.Timestamp("2017-01-05 23:59:59.999999999"),
                ),
                (
                    pd.Timestamp("2017-01-07 00:00:00"),
                    pd.Timestamp("2017-01-08 23:59:59.999999999"),
                ),
            ],
            scoring="f1",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            [
                (
                    pd.Timestamp("2017-01-05 00:00:00"),
                    pd.Timestamp("2017-01-05 23:59:59.999999999"),
                ),
                (
                    pd.Timestamp("2017-01-07 00:00:00"),
                    pd.Timestamp("2017-01-08 23:59:59.999999999"),
                ),
            ],
            scoring="recall",
        )
        == 1
    )
    assert (
        my_pipe.score(
            df,
            [
                (
                    pd.Timestamp("2017-01-05 00:00:00"),
                    pd.Timestamp("2017-01-05 23:59:59.999999999"),
                ),
                (
                    pd.Timestamp("2017-01-07 00:00:00"),
                    pd.Timestamp("2017-01-08 23:59:59.999999999"),
                ),
            ],
            scoring="iou",
        )
        == 1
    )


def test_pipenet_return_intermediate():
    """
    Test pipenet with return_intermediate=True
    """
    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    results = my_pipe.fit(df, return_intermediate=True)
    assert set(results.keys()) == set(my_pipe.steps.keys()).union({"original"})
    assert results["A-B-regression-ad"] is None
    assert results["A-C-regression-error"] is not None
    assert results["A-C-regression-ad"] is None
    assert results["ABC-ad"] is None
    assert results["D-ad"] is None
    assert results["ABCD-ad"] is None

    results = my_pipe.fit_detect(df, return_intermediate=True)
    assert set(results.keys()) == set(my_pipe.steps.keys()).union({"original"})
    pd.testing.assert_series_equal(
        results["A-B-regression-ad"],
        pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=df.index),
        check_dtype=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        results["A-C-regression-ad"],
        pd.Series([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], index=df.index),
        check_dtype=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        results["ABC-ad"],
        pd.Series([0, 0, 0, 0, 1, 0, 1, 0, 0, 0], index=df.index),
        check_dtype=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        results["D-ad"],
        pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], index=df.index),
        check_dtype=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        results["ABCD-ad"],
        pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
        check_dtype=False,
        check_names=False,
    )


def test_pipenet_return_list_return_intermediate():
    """
    Test pipenet with return_list=True and return_intermediate=True
    """
    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    results = my_pipe.fit_detect(
        df, return_list=True, return_intermediate=True
    )
    assert set(results.keys()) == set(my_pipe.steps.keys()).union({"original"})
    assert results["A-B-regression-ad"] == [
        (
            pd.Timestamp("2017-01-05 00:00:00"),
            pd.Timestamp("2017-01-05 23:59:59.999999999"),
        )
    ]
    assert results["A-C-regression-ad"] == [
        (
            pd.Timestamp("2017-01-07 00:00:00"),
            pd.Timestamp("2017-01-07 23:59:59.999999999"),
        )
    ]
    assert results["ABC-ad"] == [
        (
            pd.Timestamp("2017-01-05 00:00:00"),
            pd.Timestamp("2017-01-05 23:59:59.999999999"),
        ),
        (
            pd.Timestamp("2017-01-07 00:00:00"),
            pd.Timestamp("2017-01-07 23:59:59.999999999"),
        ),
    ]
    assert results["D-ad"] == [
        (
            pd.Timestamp("2017-01-08 00:00:00"),
            pd.Timestamp("2017-01-08 23:59:59.999999999"),
        )
    ]
    assert results["ABCD-ad"] == [
        (
            pd.Timestamp("2017-01-05 00:00:00"),
            pd.Timestamp("2017-01-05 23:59:59.999999999"),
        ),
        (
            pd.Timestamp("2017-01-07 00:00:00"),
            pd.Timestamp("2017-01-08 23:59:59.999999999"),
        ),
    ]


def test_skip_fit():
    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )
    my_pipe.fit(df)

    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )
    my_pipe.fit(df, skip_fit=["A-B-regression-ad", "A-C-regression-error"])
    assert reg_ab.coef_[0] == pytest.approx(10)
    assert reg_ac.coef_[0] == pytest.approx(100)
    assert my_pipe.steps["A-C-regression-ad"]["model"].abs_high_ == 0
    assert my_pipe.steps["A-C-regression-ad"]["model"].abs_low_ == 0

    my_pipe.fit(df, skip_fit=["A-B-regression-ad"])
    assert reg_ab.coef_[0] == pytest.approx(10)
    assert reg_ac.coef_[0] != pytest.approx(100)
    assert my_pipe.steps["A-C-regression-ad"]["model"].abs_high_ != 0
    assert my_pipe.steps["A-C-regression-ad"]["model"].abs_low_ != 0


def test_nonunique_output():
    with pytest.raises(ValueError, match="ambiguous"):
        Pipenet(
            {
                "deseasonal_residual": {
                    "model": (
                        transformer.ClassicSeasonalDecomposition(freq=6)
                    ),
                    "input": "original",
                },
                "abs_residual": {
                    "model": transformer.CustomizedTransformer1D(
                        transform_func=abs
                    ),
                    "input": "deseasonal_residual",
                },
                "iqr_ad": {
                    "model": detector.InterQuartileRangeAD(c=(None, 3)),
                    "input": "abs_residual",
                },
                "sign_check": {
                    "model": detector.ThresholdAD(high=0.0, low=-float("inf")),
                    "input": "deseasonal_residual",
                },
            }
        )


def test_transformer_pipe():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 10,
        index=pd.date_range(start="2017-1-1", periods=60, freq="D"),
    )
    my_pipe = Pipenet(
        {
            "deseasonal_residual": {
                "model": transformer.ClassicSeasonalDecomposition(freq=6),
                "input": "original",
            },
            "abs_residual": {
                "model": transformer.CustomizedTransformer1D(
                    transform_func=abs
                ),
                "input": "deseasonal_residual",
            },
        }
    )
    with pytest.raises(RuntimeError, match="`fit_transform`"):
        my_pipe.fit_detect(s)
    my_pipe.fit(s)
    my_pipe.transform(s)
    my_pipe.fit_transform(s)

    my_pipe = Pipeline(
        [
            (
                "deseasonal_residual",
                transformer.ClassicSeasonalDecomposition(freq=6),
            ),
            (
                "abs_residual",
                transformer.CustomizedTransformer1D(transform_func=abs),
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="`fit_transform`"):
        my_pipe.fit_detect(s)
    my_pipe.fit(s)
    my_pipe.transform(s)
    my_pipe.fit_transform(s)


def test_pipeline():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 5 + [0, 1, 2, 4, 2, 1] + [0, 1, 2, 3, 2, 1] * 4,
        index=pd.date_range(start="2017-1-1", periods=60, freq="D"),
    )
    my_pipe = Pipeline(
        [
            (
                "deseasonal_residual",
                transformer.ClassicSeasonalDecomposition(freq=6),
            ),
            (
                "abs_residual",
                transformer.CustomizedTransformer1D(transform_func=abs),
            ),
            ("ad", detector.QuantileAD(high=0.99)),
        ]
    )

    my_pipe.fit_detect(s)
    assert (
        my_pipe.score(
            s,
            pd.Series([0] * 33 + [1] + [0] * 26, index=s.index),
            scoring="recall",
        )
        == 1
    )
    assert (
        my_pipe.score(
            s,
            pd.Series([0] * 33 + [1] + [0] * 26, index=s.index),
            scoring="precision",
        )
        == 1
    )
    assert (
        my_pipe.score(
            s,
            pd.Series([0] * 33 + [1] + [0] * 26, index=s.index),
            scoring="f1",
        )
        == 1
    )
    assert (
        my_pipe.score(
            s,
            pd.Series([0] * 33 + [1] + [0] * 26, index=s.index),
            scoring="iou",
        )
        == 1
    )

    assert my_pipe.get_params() == {
        "deseasonal_residual": {"freq": 6, "trend": False},
        "abs_residual": {
            "fit_func": None,
            "fit_func_params": None,
            "transform_func": abs,
            "transform_func_params": None,
        },
        "ad": {"high": 0.99, "low": None},
    }


def test_pipe_summary():
    """
    Test summary
    """

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )
    my_pipe.summary()
