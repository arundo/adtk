"""Test pipeline and pipenet.

We do not test graph plot because it is covered by test_visualization."""

import pytest
import pandas as pd
import adtk.detector as detector
import adtk.aggregator as aggregator
import adtk.transformer as transformer
from adtk.pipe import Pipeline, Pipenet


def test_detector_return_intermediate():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 10,
        index=pd.date_range(start="2017-1-1", periods=60, freq="D"),
    )
    my_pipe = Pipenet(
        [
            {
                "name": "deseasonal_residual",
                "model": (transformer.ClassicSeasonalDecomposition(freq=6)),
                "input": "original",
            },
            {
                "name": "abs_residual",
                "model": transformer.CustomizedTransformer1D(
                    transform_func=abs
                ),
                "input": "deseasonal_residual",
            },
            {
                "name": "iqr_ad",
                "model": detector.InterQuartileRangeAD(c=(None, 3)),
                "input": "abs_residual",
            },
            {
                "name": "sign_check",
                "model": detector.ThresholdAD(high=0.0, low=-float("inf")),
                "input": "deseasonal_residual",
            },
            {
                "name": "and",
                "model": aggregator.AndAggregator(),
                "input": ["iqr_ad", "sign_check"],
            },
        ]
    )
    result = my_pipe.fit_detect(s, return_intermediate=True)
    assert set(result.keys()) == {
        "original",
        "deseasonal_residual",
        "abs_residual",
        "iqr_ad",
        "sign_check",
        "and",
    }


def test_skip_fit():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 10,
        index=pd.date_range(start="2017-1-1", periods=60, freq="D"),
    )

    deseasonal_residual = transformer.ClassicSeasonalDecomposition(freq=6)

    my_pipe = Pipenet(
        [
            {
                "name": "deseasonal_residual",
                "model": deseasonal_residual,
                "input": "original",
            },
            {
                "name": "abs_residual",
                "model": transformer.CustomizedTransformer1D(
                    transform_func=abs
                ),
                "input": "deseasonal_residual",
            },
            {
                "name": "iqr_ad",
                "model": detector.InterQuartileRangeAD(c=(None, 3)),
                "input": "abs_residual",
            },
            {
                "name": "sign_check",
                "model": detector.ThresholdAD(high=0.0, low=-float("inf")),
                "input": "deseasonal_residual",
            },
            {
                "name": "and",
                "model": aggregator.AndAggregator(),
                "input": ["iqr_ad", "sign_check"],
            },
        ]
    )
    with pytest.raises(RuntimeError):
        my_pipe.fit_detect(s, skip_fit=["deseasonal_residual"])
    my_pipe.fit_detect(s)


def test_nonunique_output():
    with pytest.raises(ValueError, match="ambiguous"):
        Pipenet(
            [
                {
                    "name": "deseasonal_residual",
                    "model": (
                        transformer.ClassicSeasonalDecomposition(freq=6)
                    ),
                    "input": "original",
                },
                {
                    "name": "abs_residual",
                    "model": transformer.CustomizedTransformer1D(
                        transform_func=abs
                    ),
                    "input": "deseasonal_residual",
                },
                {
                    "name": "iqr_ad",
                    "model": detector.InterQuartileRangeAD(c=(None, 3)),
                    "input": "abs_residual",
                },
                {
                    "name": "sign_check",
                    "model": detector.ThresholdAD(high=0.0, low=-float("inf")),
                    "input": "deseasonal_residual",
                },
            ]
        )


def test_transformer_pipe():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 10,
        index=pd.date_range(start="2017-1-1", periods=60, freq="D"),
    )
    my_pipe = Pipenet(
        [
            {
                "name": "deseasonal_residual",
                "model": transformer.ClassicSeasonalDecomposition(freq=6),
                "input": "original",
            },
            {
                "name": "abs_residual",
                "model": transformer.CustomizedTransformer1D(
                    transform_func=abs
                ),
                "input": "deseasonal_residual",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="`fit_transform`"):
        my_pipe.fit_detect(s)
    my_pipe.fit_transform(s)


def test_pipeline():
    s = pd.Series(
        [0, 1, 2, 3, 2, 1] * 10,
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
            ("iqr_ad", detector.InterQuartileRangeAD(c=(None, 3))),
        ]
    )

    my_pipe.fit_detect(s)
