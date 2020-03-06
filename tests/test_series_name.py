"Check if the series name or column name is correctly kept."

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

import adtk.detector as detector
import adtk.transformer as transformer
from adtk._base import _TrainableModel
from adtk._detector_base import (  # _NonTrainableMultivariateDetector,
    _NonTrainableUnivariateDetector,
    _TrainableMultivariateDetector,
    _TrainableUnivariateDetector,
)

_Detector = (
    _NonTrainableUnivariateDetector,
    # _NonTrainableMultivariateDetector,
    _TrainableUnivariateDetector,
    _TrainableMultivariateDetector,
)

# We have 4 types of models
#   - one-to-one: input a univariate series, output a univariate series
#   - one-to-many: input a univariate series, output a multivariate series
#   - many-to-one: input a multivariate series, output a univariate series
#   - many-to-many: input a multivariate series, output a multivariate series

one2one_models = [
    detector.ThresholdAD(),
    detector.QuantileAD(),
    detector.InterQuartileRangeAD(),
    detector.GeneralizedESDTestAD(),
    detector.PersistAD(window=10),
    detector.LevelShiftAD(window=10),
    detector.VolatilityShiftAD(window=10),
    detector.AutoregressionAD(),
    detector.SeasonalAD(freq=2),
    transformer.RollingAggregate(window=10, agg="median"),
    transformer.RollingAggregate(
        window=10, agg="quantile", agg_params={"q": 0.5}
    ),
    transformer.DoubleRollingAggregate(window=10, agg="median"),
    transformer.DoubleRollingAggregate(
        window=10, agg="quantile", agg_params={"q": [0.1, 0.5, 0.9]}
    ),
    transformer.DoubleRollingAggregate(
        window=10, agg="hist", agg_params={"bins": [30, 50, 70]}
    ),
    transformer.StandardScale(),
    transformer.ClassicSeasonalDecomposition(freq=2),
]

one2many_models = [
    transformer.RollingAggregate(
        window=10, agg="quantile", agg_params={"q": [0.1, 0.5, 0.9]}
    ),
    transformer.RollingAggregate(
        window=10, agg="hist", agg_params={"bins": [20, 50, 80]}
    ),
    transformer.Retrospect(n_steps=3),
]

many2one_models = [
    detector.MinClusterDetector(KMeans(n_clusters=2)),
    detector.OutlierDetector(
        LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    ),
    detector.RegressionAD(target="A", regressor=LinearRegression()),
    detector.PcaAD(),
    transformer.SumAll(),
    transformer.RegressionResidual(target="A", regressor=LinearRegression()),
    transformer.PcaReconstructionError(),
]


@pytest.mark.parametrize("model", one2one_models)
def test_one2one_s2s_w_name(model):
    """
    if a one-to-one model is applied to a Series, it should keep the Series
    name unchanged
    """
    s_name = pd.Series(
        np.arange(100),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        name="A",
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(s_name)
    else:
        result = model.predict(s_name)
    assert result.name == "A"


@pytest.mark.parametrize("model", one2one_models)
def test_one2one_s2s_wo_name(model):
    """
    if a one-to-one model is applied to a Series, it should keep the Series
    name unchanged
    """
    s_no_name = pd.Series(
        np.arange(100),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(s_no_name)
    else:
        result = model.predict(s_no_name)
    assert result.name is None


@pytest.mark.parametrize("model", one2one_models)
def test_one2one_df2df(model):
    """
    if a one-to-one model is applied to a DataFrame, it should keep the column
    names unchanged
    """
    df = pd.DataFrame(
        np.arange(300).reshape(100, 3),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(df)
    else:
        result = model.predict(df)
    assert list(result.columns) == ["A", "B", "C"]


@pytest.mark.parametrize("model", one2one_models)
def test_one2one_df2list(model):
    """
    if a one-to-one model (detector) is applied to a DataFrame and returns a
    dict, the output dict keys should match the input column names
    """
    if isinstance(model, _Detector):
        df = pd.DataFrame(
            np.arange(300).reshape(100, 3),
            index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
            columns=["A", "B", "C"],
        )
        if isinstance(model, _TrainableModel):
            result = model.fit_detect(df, return_list=True)
        else:
            result = model.detect(df, return_list=True)
        if sys.version_info[1] >= 6:
            assert list(result.keys()) == ["A", "B", "C"]
        else:
            assert set(result.keys()) == {"A", "B", "C"}


@pytest.mark.parametrize("model", one2many_models)
def test_one2many_s2df_w_name(model):
    """
    if a one-to-many model is applied to a Series, the output should not have
    prefix in column names, no matter whether the input Series has a name.
    """
    s_name = pd.Series(
        np.arange(100),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        name="A",
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(s_name)
    else:
        result = model.predict(s_name)
    assert all([col[:2] != "A_" for col in result.columns])


@pytest.mark.parametrize("model", one2many_models)
def test_one2many_s2df_wo_name(model):
    """
    if a one-to-many model is applied to a Series, the output should not have
    prefix in column names, no matter whether the input Series has a name.
    """
    s_no_name = pd.Series(
        np.arange(100),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(s_no_name)
    else:
        result = model.predict(s_no_name)
    assert all([col[:2] != "A_" for col in result.columns])


@pytest.mark.parametrize("model", one2many_models)
def test_one2many_df2df(model):
    """
    if a one-to-many model is applied to a DataFrame, the output should have
    prefix in column names to indicate the input columns they correspond.
    """
    df = pd.DataFrame(
        np.arange(300).reshape(100, 3),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(df)
    else:
        result = model.predict(df)
    n_cols = round(len(result.columns) / 3)
    assert all([col[:2] == "A_" for col in result.columns[:n_cols]])
    assert all([col[2:4] != "A_" for col in result.columns[:n_cols]])
    assert all(
        [col[:2] == "B_" for col in result.columns[n_cols : 2 * n_cols]]
    )
    assert all(
        [col[2:4] != "B_" for col in result.columns[n_cols : 2 * n_cols]]
    )
    assert all([col[:2] == "C_" for col in result.columns[2 * n_cols :]])
    assert all([col[2:4] != "C_" for col in result.columns[2 * n_cols :]])


@pytest.mark.parametrize("model", many2one_models)
def test_many2one(model):
    """
    The output Series from a many-to-one model should NOT have name
    """
    df = pd.DataFrame(
        np.arange(300).reshape(100, 3),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    if isinstance(model, _TrainableModel):
        result = model.fit_predict(df)
    else:
        result = model.predict(df)
    assert result.name is None


def test_pca_reconstruction():
    df = pd.DataFrame(
        np.arange(300).reshape(100, 3),
        index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    result = transformer.PcaReconstruction(k=2).fit_predict(df)
    assert list(result.columns) == ["A", "B", "C"]
